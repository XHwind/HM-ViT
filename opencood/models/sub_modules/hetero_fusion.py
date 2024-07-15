import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce

from opencood.models.base_transformer import HeteroPreNorm, HeteroFeedForward, \
    HeteroLayerNorm, HeteroPreNormResidual
from opencood.models.sub_modules.torch_transformation_utils import \
    get_roi_and_cav_mask
from opencood.models.sub_modules.spatial_transformation import \
    SpatialTransformation
from opencood.models.fusion_modules.split_attn import SplitAttn

# swap attention -> max_vit
class HeteroAttention(nn.Module):
    """
    Unit Attention class.

    Parameters
    ----------
    dim: int
        Input feature dimension.
    dim_head: int
        The head dimension.
    dropout: float
        Dropout rate
    agent_size: int
        The agent can be different views, timestamps or vehicles.
    """

    def __init__(
            self,
            dim,
            dim_head=32,
            dropout=0.,
            agent_size=6,
            window_size=7,
            num_types=2
    ):
        super().__init__()
        assert (dim % dim_head) == 0, \
            'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.num_types = num_types
        self.use_position_emb = True
        num_relations = num_types ** 2
        self.window_size = [agent_size, window_size, window_size]

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1)
        )

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        for t in range(num_types):
            self.k_linears.append(nn.Linear(dim, dim))
            self.q_linears.append(nn.Linear(dim, dim))
            self.v_linears.append(nn.Linear(dim, dim))
            self.a_linears.append(
                nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout)))
        # self.to_qkv = HeteroFeedForward(dim, 3 * dim, out_dim=3 * dim)

        self.relation_att = nn.Parameter(
            torch.Tensor(num_relations,
                         self.heads,
                         dim_head, dim_head))
        self.relation_msg = nn.Parameter(
            torch.Tensor(num_relations,
                         self.heads,
                         dim_head, dim_head))
        if self.use_position_emb:
            self.build_positional_embedding()
        torch.nn.init.xavier_uniform(self.relation_att)
        torch.nn.init.xavier_uniform(self.relation_msg)

    def build_positional_embedding(self):
        Wh, Ww = self.window_size[1], self.window_size[2]
        self.relative_position_bias_table = nn.Embedding(
            (2 * Wh - 1) *
            (2 * Ww - 1),
            self.heads)  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for
        # each token inside the window
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        # 2, Wh, Ww
        coords = torch.stack(torch.meshgrid(coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)  # 2, Wd*Wh*Ww

        # 2, Wh*Ww, Wh*Ww
        relative_coords = \
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # Wh*Ww, Wh*Ww, 2
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += Wh - 1
        relative_coords[:, :, 1] += Ww - 1

        relative_coords[:, :, 0] *= (2 * Wh - 1)
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

    def to_qkv(self, x, types):
        # x: (b, l, h, w, w_h, w_w, c)
        # types: (B,L)
        B, L = x.shape[0], x.shape[1]
        q_batch = []
        k_batch = []
        v_batch = []

        for b in range(B):
            q_list = []
            k_list = []
            v_list = []

            for i in range(L):
                # (h, w, w_h, w_w, c)
                q_list.append(
                    self.q_linears[types[b, i]](x[b, i, ...]))
                k_list.append(
                    self.k_linears[types[b, i]](x[b, i, ...]))
                v_list.append(
                    self.v_linears[types[b, i]](x[b, i, ...]))
            # (l, h, w, w_h, w_w, c)
            q_batch.append(torch.stack(q_list, dim=0))
            k_batch.append(torch.stack(k_list, dim=0))
            v_batch.append(torch.stack(v_list, dim=0))
        # x: (b, l, h, w, w_h, w_w, c)
        q = torch.stack(q_batch, dim=0)
        k = torch.stack(k_batch, dim=0)
        v = torch.stack(v_batch, dim=0)
        return q, k, v

    def to_out(self, x, types):
        # x: (b l x y w1 w2 C)
        # types: (b, l)
        out_batch = []
        for b in range(x.shape[0]):
            out_list = []
            for i in range(x.shape[1]):
                out_list.append(self.a_linears[types[b, i]](x[b, i, ...]))
            out_batch.append(torch.stack(out_list, dim=0))
        out = torch.stack(out_batch, dim=0)
        return out

    def get_relation_type_index(self, type1, type2):
        return type1 * self.num_types + type2

    def get_hetero_edge_weights(self, x, types):
        w_att_batch = []
        w_msg_batch = []
        B, L, xx, yy = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        window_size = self.window_size[1]
        for b in range(B):
            w_att_list = []
            w_msg_list = []

            for i in range(L):
                w_att_i_list = []
                w_msg_i_list = []

                for j in range(L):
                    e_type = self.get_relation_type_index(types[b, i],
                                                          types[b, j])
                    w_att_i_list.append(self.relation_att[e_type].unsqueeze(0))
                    w_msg_i_list.append(self.relation_msg[e_type].unsqueeze(0))
                w_att_list.append(torch.cat(w_att_i_list, dim=0).unsqueeze(0))
                w_msg_list.append(torch.cat(w_msg_i_list, dim=0).unsqueeze(0))

            w_att_batch.append(torch.cat(w_att_list, dim=0).unsqueeze(0))
            w_msg_batch.append(torch.cat(w_msg_list, dim=0).unsqueeze(0))

        # (B,M,L,L,C_head,C_head)
        # (B, L, L, M, C_head, C_head)
        w_att = torch.cat(w_att_batch, dim=0)
        w_msg = torch.cat(w_msg_batch, dim=0)
        return w_att, w_msg

    def forward(self, x, mode, mask=None, exclude_self=False):
        # x shape: b, l, h, w, w_h, w_w, c
        batch, agent_size, height, width, window_height, window_width, _, device, h \
            = *x.shape, x.device, self.heads

        # project for queries, keys, values
        q, k, v = self.to_qkv(x, mode)
        # q, k, v = self.to_qkv(x, mode).chunk(3, dim=-1)
        # split heads
        q, k, v = map(
            lambda t: rearrange(t, 'b l x y w1 w2 (h d) -> b x y h l w1 w2 d',
                                h=h),
            (q, k, v))
        q = q[:, :, :, :, :1, :, :, :]
        # exlude self-attention
        if exclude_self:
            k = k[:, :, :, :, 1:, :, :, :]
            v = v[:, :, :, :, 1:, :, :, :]
            mask = mask[..., 1:]

        # (B, L, L, M, C_head, C_head)
        w_att, w_msg = self.get_hetero_edge_weights(x, mode)
        w_att = w_att[:, :1, :, :, :, :]
        w_msg = w_msg[:, :1, :, :, :, :]

        if exclude_self:
            w_att = w_att[:, :, 1:, ...]
            w_msg = w_msg[:, :, 1:, ...]

        # scale
        q = q * self.scale
        # for q: l w1 w2 -> a c d
        # for k: l w1 w2 -> z e f
        # (b , x, y, M, L1, w11, w12, L2, w21, w22)
        sim = einsum(
            'b x y h a c d p, b a z h p q, b x y h z e f q -> b x y h a c d z e f',
            [q, w_att, k])
        sim_shape = sim.shape

        # add relative positional embedding
        if self.use_position_emb:
            bias = self.relative_position_bias_table(
                self.relative_position_index)
            # pad non position directions
            bias = rearrange(bias, 'i j h -> h i j')[(None,) * 5]
            sim = rearrange(sim,
                            'b x y h a c d z e f -> b x y a z h (c d) (e f)') + bias
            sim = rearrange(sim,
                            ' b x y a z h (c d) (e f) -> (b x y) h (a c d) (z e f)',
                            c=sim_shape[5], d=sim_shape[6],
                            e=sim_shape[8], f=sim_shape[9])
        else:
            sim = rearrange(sim,
                            'b x y h a c d z e f -> (b x y) h (a c d) (z e f)')

        # mask shape if exist: b x y w1 w2 e l
        if mask is not None:
            # b x y w1 w2 e l -> (b x y) 1 (l w1 w2)
            mask = rearrange(mask, 'b x y w1 w2 e l -> (b x y) e (l w1 w2)')
            # (b x y) 1 1 (l w1 w2) = b h 1 n
            mask = mask.unsqueeze(1)
            sim = sim.masked_fill(mask == 0, -float('inf'))

        # attention
        attn = self.attend(sim)
        if exclude_self:
            attn = torch.where(attn == attn, attn,
                               torch.tensor(0., dtype=attn.dtype,
                                            device=attn.device))
        attn = rearrange(attn,
                         '(b x y) h (a c d) (z e f) -> b x y h a c d z e f',
                         b=sim_shape[0], x=sim_shape[1], y=sim_shape[2],
                         h=sim_shape[3], a=sim_shape[4], c=sim_shape[5],
                         d=sim_shape[6], z=sim_shape[7], e=sim_shape[8],
                         f=sim_shape[9])

        v_msg = einsum('b a z h p q, b x y h z e f p -> b x y h a z e f q',
                       [w_msg, v])
        out = torch.einsum(
            'b x y h a c d z e f, b x y h a z e f q->b x y h a c d q',
            [attn, v_msg])

        # out = torch.einsum('b x y h a c d z e f, b a z h p q, b x y h z e f q->b x y h a c d q', [attn, w_msg, v])

        # out = torch.einsum(
        #     'b x y h a c d z e f, b x y h z e f q-> b x y h a c d q',
        #     [attn, v])
        out = rearrange(out, 'b x y h a c d q->b a x y c d (h q)')
        out = self.to_out(out, mode)

        return out

class HeteroFusionBlock(nn.Module):
    """
    Swap Fusion Block contains window attention and grid attention with
    mask enabled for multi-vehicle cooperation.
    """

    def __init__(self, config):
        super(HeteroFusionBlock, self).__init__()
        input_dim = config['input_dim']
        mlp_dim = config['mlp_dim']
        agent_size = config['agent_size']
        window_size = config['window_size']
        drop_out = config['drop_out']
        dim_head = config['dim_head']
        self.architect_mode = config['architect_mode']
        if self.architect_mode == 'parallel':
            self.split_attn = SplitAttn(input_dim, num_windows=2)

        self.spatial_transform = SpatialTransformation(
            config['spatial_transform'])
        self.downsample_rate = config['spatial_transform']['downsample_rate']
        self.discrete_ratio = config['spatial_transform']['voxel_size'][0]

        self.window_size = window_size

        self.window_norm = HeteroLayerNorm(input_dim)

        self.window_attention = HeteroAttention(input_dim,
                                                dim_head,
                                                drop_out,
                                                agent_size,
                                                window_size)

        self.window_ffd = HeteroPreNormResidual(input_dim,
                                        HeteroFeedForward(input_dim, mlp_dim,
                                                          drop_out))

        self.grid_norm = HeteroLayerNorm(input_dim)

        self.grid_attention = HeteroAttention(input_dim,
                                              dim_head,
                                              drop_out,
                                              agent_size,
                                              window_size)
        self.grid_ffd = HeteroPreNormResidual(input_dim,
                                      HeteroFeedForward(input_dim, mlp_dim,
                                                        drop_out))
        self.aggregate_fc = HeteroFeedForward(mlp_dim * 3, mlp_dim, drop_out,
                                              out_dim=mlp_dim)

    def change_ith_to_first(self, x_pair, mask_pair, mode, i):
        """"""
        L = x_pair.shape[1]
        order = [i] + [j for j in range(L) if j != i]
        x_agent = x_pair[:, order, ...]
        mask_agent = mask_pair[:, :, :, :, order]
        mode_agent = mode[:, order]
        return x_agent, mask_agent, mode_agent

    def warp_features(self, x, pairwise_t_matrix, mask):
        # x: (B, L, C, H, W)
        # pairwise_t_matrix: (B, L, L, 4, 4)
        B, L, C, H, W = x.shape
        x_pair = []
        mask_pair = []
        for i in range(L):
            transformation_matrix = pairwise_t_matrix[:, :, i, :, :]
            x_agent = self.spatial_transform(x, transformation_matrix)
            # (B, H, W, 1, L)
            com_mask = get_roi_and_cav_mask(
                (B, L, H, W, C),
                mask,
                transformation_matrix,
                self.discrete_ratio,
                self.downsample_rate)
            x_pair.append(x_agent)
            mask_pair.append(com_mask)
        # (B, L, L, C, H, W)
        x_pair = torch.stack(x_pair, dim=2)
        # (B, H, W, 1, L, L)
        # mask[...,i,j]: i->j mask
        mask_pair = torch.stack(mask_pair, dim=-1)
        return x_pair, mask_pair

    def local_spatial_multi_agent_attention(self, x, pairwise_t_matrix, mask,
                                            mode, record_len, exclude_self):
        # x: b l c h w
        # mask: b h w 1 l
        # window attention -> grid attention
        x_normed = self.window_norm(x.permute(0, 1, 3, 4, 2), mode).permute(0, 1, 4,
                                                                      2, 3)
        x_pair, mask_pair = self.warp_features(x_normed, pairwise_t_matrix,
                                               mask)

        max_cav = record_len.max()
        B, L = pairwise_t_matrix.shape[:2]
        x_updated = []
        for i in range(max_cav):
            x_agent, mask_agent, mode_agent = self.change_ith_to_first(
                x_pair[:, :max_cav, i, ...],
                mask_pair[:, :, :, :, :max_cav, i],
                mode[:, :max_cav],
                i)
            mask_swap = mask_agent
            # mask b h w 1 l -> b x y w1 w2 1 L
            mask_swap = rearrange(mask_swap,
                                  'b (x w1) (y w2) e l -> b x y w1 w2 e l',
                                  w1=self.window_size, w2=self.window_size)
            x_agent = rearrange(x_agent,
                                'b m d (x w1) (y w2) -> b m x y w1 w2 d',
                                w1=self.window_size, w2=self.window_size)

            x_agent = self.window_attention(x_agent, mode_agent, mask=mask_swap,
                                            exclude_self=exclude_self)
            x_agent = rearrange(x_agent,
                                'b m x y w1 w2 d -> b m d (x w1) (y w2)')

            x_updated.append(x_agent)
        x_updated = torch.cat(x_updated, dim=1)
        # (B, L, C, H, W)
        x = F.pad(x_updated, (0, 0, 0, 0, 0, 0, 0, L - max_cav, 0, 0)) + x

        x = self.window_ffd(x.permute(0, 1, 3, 4, 2), mode).permute(0, 1, 4, 2,
                                                                    3)

        return x

    def global_spatial_multi_agent_attention(self, x, pairwise_t_matrix, mask,
                                             mode, record_len, exclude_self):
        # grid attention
        # x: b l c h w
        # mask: b h w 1 l
        # window attention -> grid attention
        x_normed = self.grid_norm(x.permute(0, 1, 3, 4, 2), mode).permute(0, 1, 4,
                                                                    2, 3)
        x_pair, mask_pair = self.warp_features(x_normed, pairwise_t_matrix,
                                               mask)

        max_cav = record_len.max()
        B, L = pairwise_t_matrix.shape[:2]
        x_updated = []
        for i in range(max_cav):
            x_agent, mask_agent, mode_agent = self.change_ith_to_first(
                x_pair[:, :max_cav, i, ...],
                mask_pair[:, :, :, :, :max_cav, i],
                mode[:, :max_cav],
                i)
            mask_swap = mask_agent
            mask_swap = rearrange(mask_swap,
                                  'b (w1 x) (w2 y) e l -> b x y w1 w2 e l',
                                  w1=self.window_size, w2=self.window_size)
            x_agent = rearrange(x_agent, 'b m d (w1 x) (w2 y) -> b m x y w1 w2 d',
                          w1=self.window_size, w2=self.window_size)
            x_agent = self.grid_attention(x_agent, mode_agent, mask=mask_swap,
                                    exclude_self=exclude_self)
            x_agent = rearrange(x_agent, 'b m x y w1 w2 d -> b m d (w1 x) (w2 y)')

            x_updated.append(x_agent)
        x_updated = torch.cat(x_updated, dim=1)
        # (B, L, C, H, W)
        x = F.pad(x_updated, (0, 0, 0, 0, 0, 0, 0, L - max_cav, 0, 0)) + x

        x = self.grid_ffd(x.permute(0, 1, 3, 4, 2), mode).permute(0, 1, 4, 2,
                                                                  3)

        return x

    def forward(self, x, pairwise_t_matrix, mode, record_len, mask):
        # x: (B, L, C, H, W)
        # pairwise_t_matrix: (B, L, L, 4, 4)
        # mask: (B, H, W, 1, L, L)
        if self.architect_mode == 'sequential':
            x = self.local_spatial_multi_agent_attention(x, pairwise_t_matrix,
                                                         mask,
                                                         mode, record_len,
                                                         exclude_self=False)
            x = self.global_spatial_multi_agent_attention(x, pairwise_t_matrix,
                                                          mask,
                                                          mode, record_len,
                                                          exclude_self=False)
        elif self.architect_mode == 'parallel':
            x_local = self.local_spatial_multi_agent_attention(x, pairwise_t_matrix,
                                                         mask,
                                                         mode, record_len,
                                                         exclude_self=False)
            x_global = self.global_spatial_multi_agent_attention(x, pairwise_t_matrix,
                                                          mask,
                                                          mode, record_len,
                                                          exclude_self=False)
            x = self.split_attn([x_local.permute(0,1,3,4,2),
                                 x_global.permute(0,1,3,4,2)])
            x = x.permute(0,1,4,2,3)
        else:
            raise ValueError(f"{self.architect_mode} not implemented")

        return x


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args = {'input_dim': 512,
            'mlp_dim': 512,
            'agent_size': 4,
            'window_size': 8,
            'dim_head': 4,
            'drop_out': 0.1,
            'depth': 2,
            'mask': True
            }
    block = HeteroFusionBlock(args)
    block.cuda()
    test_data = torch.rand(1, 4, 512, 32, 32)
    test_data = test_data.cuda()
    mask = torch.ones(1, 32, 32, 1, 4)
    mask = mask.cuda()

    output = block(test_data, mask)
    print(output)
