import torch
from torch import nn
from einops import rearrange

from opencood.models.base_transformer import FeedForward, PreNorm
from opencood.models.backbones.resnet_ms import ResnetEncoder
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.sub_modules.bev_seg_head import BevSegHead


class ViewProjectionModule(nn.Module):
    def __init__(self, dim, mlp_dim, dropout):
        super(ViewProjectionModule, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Project the frontal representation to bev view.

        Parameters
        ----------
        x : torch.Tensor
            The frontal image feature from certain camera.
            Shape: (B, L, C, H, W)

        Returns
        -------
        The camera's feature projected to bev view.
        """
        b, l, c, h, w = x.shape

        # b, l, c, hw
        x = rearrange(x, 'b l c h w -> b l c (h w)')
        # projection to bev
        x = self.net(x)
        # b, l, c, hw -> b, l, c, h, w
        x = rearrange(x, 'b l c (h w) -> b l c h w',
                      h=h, w=w)
        return x


class ViewAttentionModule(nn.Module):
    """
    Basic attention module to fuse all views from different cameras together.
    """

    def __init__(self, dim, heads, dim_head=64, dropout=0.1):
        super(ViewAttentionModule, self).__init__()

        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Perform self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Features of all images per agent. Shape: (B, L, H, W, M, C)

        Returns
        -------
        A tensor of shape (B, L, M, H, W, C)
        """
        # B L M H W C -> B L H W M C
        x = x.permute(0, 1, 3, 4, 2, 5)
        # qkv: [(B, L, H, W, M, C_inner) *3]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # q: (B, L, D, H, W, M, C)
        q, k, v = map(
            lambda t: rearrange(t, 'b l h w m (d c) -> b l d h w m c',
                                d=self.heads), qkv)
        # attention, (B, L, D, H, W, M, M)
        att_map = torch.einsum('b l d h w i c, b l d h w j c -> b l d h w i j',
                               q, k) * self.scale
        # softmax
        att_map = self.attend(att_map)
        # out:(B, L, D, H, W, M, C_head)
        out = torch.einsum('b l d h w i j, b l d h w j c -> b l d h w i c',
                           att_map, v)
        out = rearrange(out, 'b l d h w m c -> b l h w m (d c)',
                        d=self.heads)
        out = self.to_out(out)
        # (B L H W M C) -> (B L M H W C)
        out = out.permute(0, 1, 4, 2, 3, 5)

        return out


class ViewTransferModule(nn.Module):
    """
    View parser module
    """

    def __init__(self, args):
        super(ViewTransferModule, self).__init__()

        input_dim = args['dim']

        # view projection module
        spatial_dim = args['vpm']['dim']
        dropout_vpm = args['vpm']['dropout']
        hidden_dim = args['vpm']['hidden_dim']
        num_cam = args['vpm']['num_cam']
        vpm_depth = args['vpm']['depth']

        self.vpms = nn.ModuleList([])
        for _ in range(vpm_depth):
            vpm_cameras = nn.ModuleList([])
            for _ in range(num_cam):
                vpm_cameras.append(ViewProjectionModule(spatial_dim,
                                                        hidden_dim,
                                                        dropout_vpm))
            self.vpms.append(vpm_cameras)

        # view attention module
        self.vam_layers = nn.ModuleList([])
        heads = args['vam']['heads']
        dim_head = args['vam']['dim_head']
        dropout_vam = args['vam']['dropout']
        vam_depth = args['vam']['depth']

        # feed forward MLP params
        mlp_dim = args['feed_forward']['mlp_dim']
        dropout_ffw = args['feed_forward']['dropout']

        for _ in range(vam_depth):
            self.vam_layers.append(nn.ModuleList([
                PreNorm(input_dim, ViewAttentionModule(dim=input_dim,
                                                       heads=heads,
                                                       dim_head=dim_head,
                                                       dropout=dropout_vam)),
                PreNorm(input_dim, FeedForward(input_dim,
                                               mlp_dim,
                                               dropout=dropout_ffw))
            ]))

    def forward(self, x):
        """
        Perform view projection first and then fuse all projected views
        via transformer.

        Parameters
        ----------
        x : torch.Tensor
            Resent features for all camera images,
            shape (B, L, M, C, H, W)

        Returns
        -------
        The fused bev feature with shape of (B, L, C, H, W)
        """
        b, l, m, c, h, w = x.shape
        outputs = x.clone()

        # view projection for each camera
        for vpm in self.vpms:
            for i in range(m):
                single_cam = outputs[:, :, i]
                projected_cam = vpm[i](single_cam)
                outputs[:, :, i] = projected_cam

        x = outputs
        # x: (B, L, M, C, H, W) -> (B, L, M, H, W, C)
        x = x.permute(0, 1, 2, 4, 5, 3)
        # view transformer fusion
        for attn, ff in self.vam_layers:
            x = attn(x) + x
            x = ff(x) + x

        # mean on m dimension
        x = torch.mean(x, dim=2)
        # B L C H W
        x = x.permute(0, 1, 4, 2, 3)
        return x


class ViewParseNetwork(nn.Module):
    """
    Encoder + ViewProjection + ViewFusion Stating.

    Parameters
    __________
    params: dict
        Parameters of all sub models.
    """

    def __init__(self, params):
        super(ViewParseNetwork, self).__init__()

        # encoder params
        encoder_params = params['encoder']
        self.encoder = ResnetEncoder(encoder_params)
        self.conv_1x1_param = None

        if 'conv1x1' in params:
            self.conv_1x1_param = params['conv1x1']
            self.conv1_x1 = nn.Conv2d(self.conv_1x1_param['input_dim'],
                                      self.conv_1x1_param['output_dim'],
                                      kernel_size=1)

        # view parse module
        self.vpm = ViewTransferModule(params['vtm'])

        # decoder
        self.decoder = NaiveDecoder(params['decoder'])

        # segmentation head
        self.target = params['target']
        self.seg_head = BevSegHead(self.target,
                                   params['seg_head_dim'],
                                   params['output_class'])

    def forward(self, batch_dict):
        # the input can be a dictionary or a tensor.
        if isinstance(batch_dict, dict):
            # shape: (B,L,M,H,W,3)
            x = batch_dict['inputs']
        else:
            # shape: (B,L,M,H,W,3)
            x = batch_dict

        x = self.encoder(x)
        if self.conv_1x1_param is not None:
            b, l, m, _, _, _ = x.shape
            x = rearrange(x, 'b l m c h w -> (b l m) c h w')
            x = self.conv1_x1(x)
            x = rearrange(x, '(b l m) c h w -> b l m c h w',
                          b=b, l=l, m=m)
        x = self.vpm(x)
        x = self.decoder(x)

        # reshape to correct format
        b, l, c, h, w = x.shape
        x = rearrange(x, 'b l c h w -> (b l) c h w')

        output_dict = self.seg_head(x, b, l)
        return output_dict


if __name__ == '__main__':
    import os
    from opencood.hypes_yaml.yaml_utils import load_yaml

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    test_data = torch.rand(4, 1, 4, 512, 512, 3)
    test_data = test_data.cuda()

    params = load_yaml('/home/runshengxu/project/opencood_camera/'
                       'opencood/hypes_yaml/opcamera/view_parse_network_static.yaml')

    model = ViewParseNetwork(params['model']['args'])
    model = model.cuda()
    while True:
        output = model({'inputs': test_data})
        print('test_passed')
