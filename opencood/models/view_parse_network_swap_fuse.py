import torch
from torch import nn
from einops import rearrange

from opencood.models.view_parse_network import ViewTransferModule
from opencood.models.backbones.resnet_ms import ResnetEncoder
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.fusion_modules.swap_fusion_modules import \
    SwapFusionEncoder
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    get_transformation_matrix, warp_affine, get_roi_and_cav_mask, \
    get_discretized_transformation_matrix
from opencood.models.sub_modules.bev_seg_head import BevSegHead


class STTF(nn.Module):
    def __init__(self, args):
        super(STTF, self).__init__()
        self.discrete_ratio = args['resolution']
        self.downsample_rate = args['downsample_rate']

    def forward(self, x, spatial_correction_matrix):
        """
        Transform the bev features to ego space.

        Parameters
        ----------
        x : torch.Tensor
            B L C H W
        spatial_correction_matrix : torch.Tensor
            Transformation matrix to ego

        Returns
        -------
        The bev feature same shape as x but with transformation
        """
        dist_correction_matrix = get_discretized_transformation_matrix(
            spatial_correction_matrix, self.discrete_ratio,
            self.downsample_rate)

        # transpose and flip to make the transformation correct
        x = rearrange(x, 'b l c h w  -> b l c w h')
        x = torch.flip(x, dims=(4,))
        # Only compensate non-ego vehicles
        B, L, C, H, W = x.shape

        T = get_transformation_matrix(
            dist_correction_matrix[:, :, :, :].reshape(-1, 2, 3), (H, W))
        cav_features = warp_affine(x[:, :, :, :, :].reshape(-1, C, H, W), T,
                                   (H, W))
        cav_features = cav_features.reshape(B, -1, C, H, W)

        # flip and transpose back
        x = cav_features
        x = torch.flip(x, dims=(4, ))
        x = rearrange(x, 'b l c w h -> b l h w c')

        return x


class ViewParseNetworkSwapFuse(nn.Module):
    """
    Encoder + ViewProjection + ViewFusion Stating.

    Parameters
    __________
    params: dict
        Parameters of all sub models.
    """

    def __init__(self, params):
        super(ViewParseNetworkSwapFuse, self).__init__()

        self.max_cav = params['max_cav']
        # encoder params
        encoder_params = params['encoder']
        self.encoder = ResnetEncoder(encoder_params)

        # view parse module
        self.vpm = ViewTransferModule(params['vtm'])

        # spatial feature transform module
        self.downsample_rate = params['sttf']['downsample_rate']
        self.discrete_ratio = params['sttf']['resolution']
        self.use_roi_mask = params['sttf']['use_roi_mask']
        self.sttf = STTF(params['sttf'])

        # spatial fusion
        self.fusion_net = SwapFusionEncoder(params['swap_fusion'])

        # decoder
        self.decoder = NaiveDecoder(params['decoder'])

        # segmentation head
        self.target = params['target']
        self.seg_head = BevSegHead(self.target,
                                   params['seg_head_dim'],
                                   params['output_class'])

    def forward(self, batch_dict):
        # shape: (B*L, 1, M, H, W, 3)
        x = batch_dict['inputs']
        # shape: (B, max_cav, 4, 4)
        transformation_matrix = batch_dict['transformation_matrix']
        record_len = batch_dict['record_len']

        # B*L, 1, M, C, H, W
        x = self.encoder(x)
        # B*L, 1, C, H, W
        x = self.vpm(x)

        # B*L, C, H, W
        x = x.squeeze(1)
        # Reformat to (B, max_cav, C, H, W)
        x, mask = regroup(x, record_len, self.max_cav)
        # perform feature spatial transformation,  B, max_cav, H, W, C
        x = self.sttf(x, transformation_matrix)
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(
            3) if not self.use_roi_mask \
            else get_roi_and_cav_mask(x.shape,
                                      mask,
                                      transformation_matrix,
                                      self.discrete_ratio,
                                      self.downsample_rate)

        # fuse all agents together to get a single bev map, b h w c
        x = rearrange(x, 'b l h w c -> b l c h w')
        x = self.fusion_net(x, com_mask)
        x = x.unsqueeze(1)

        # decode to the right size
        x = self.decoder(x)

        # reshape to correct format
        b, l, c, h, w = x.shape
        x = rearrange(x, 'b l c h w -> (b l) c h w')

        output_dict = self.seg_head(x, b, l)
        return output_dict
