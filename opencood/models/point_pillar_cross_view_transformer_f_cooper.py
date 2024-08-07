"""
Implementation of Brady Zhou's cross view transformer
"""
import einops
import numpy as np
import torch.nn as nn
import torch
from einops import rearrange

from opencood.models.sub_modules.fax_modules import FAXModule
from opencood.models.backbones.resnet_ms import ResnetEncoder
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.sub_modules.bev_seg_head import BevSegHead
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fusion_modules.swap_fusion_modules import \
    SwapFusionEncoder
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    get_transformation_matrix, warp_affine, get_roi_and_cav_mask, \
    get_discretized_transformation_matrix
from opencood.models.fusion_modules.f_cooper_fuse import SpatialFusionMask

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
        x = torch.flip(x, dims=(4,))
        x = rearrange(x, 'b l c w h -> b l h w c')

        return x


class PointPillarCrossViewTransformerFCooper(nn.Module):
    def __init__(self, config):
        super(PointPillarCrossViewTransformerFCooper, self).__init__()
        self.max_cav = config['max_cav']
        # encoder params
        self.encoder = ResnetEncoder(config['encoder'])

        # cvm params
        fax_params = config['fax']
        fax_params['backbone_output_shape'] = self.encoder.output_shapes
        self.fax = FAXModule(fax_params)

        if config['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(128, config['compression'])
        else:
            self.compression = False

        # spatial feature transform module
        self.downsample_rate = config['sttf']['downsample_rate']
        self.discrete_ratio = config['sttf']['resolution']
        self.use_roi_mask = config['sttf']['use_roi_mask']
        self.sttf = STTF(config['sttf'])

        # spatial fusion
        self.fusion_net = SpatialFusionMask()

        # decoder params
        decoder_params = config['decoder']
        # decoder for dynamic and static differet
        self.decoder = NaiveDecoder(decoder_params)

        self.cls_head = nn.Conv2d(32, config['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(32, 7 * config['anchor_number'],
                                  kernel_size=1)

    def forward(self, batch_dict):
        batch_dict['camera'] = batch_dict['camera'].unsqueeze(1)
        batch_dict['intrinsic'] = batch_dict['intrinsic'].unsqueeze(1)
        batch_dict['extrinsic'] = batch_dict['extrinsic'].unsqueeze(1)
        x = batch_dict['camera']
        b, l, m, _, _, _ = x.shape

        # shape: (B, max_cav, 4, 4)
        transformation_matrix = batch_dict['transformation_matrix']
        record_len = batch_dict['record_len']

        x = self.encoder(x)
        batch_dict.update({'features': x})
        x = self.fax(batch_dict)

        # B*L, C, H, W
        x = x.squeeze(1)

        # compressor
        if self.compression:
            x = self.naive_compressor(x)

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
        x = self.fusion_net(x)
        x = x.unsqueeze(1)

        # dynamic head
        x = self.decoder(x)
        x = rearrange(x, 'b l c h w -> (b l) c h w')

        psm = self.cls_head(x)
        rm = self.reg_head(x)
        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict


if __name__ == '__main__':
    import os
    import torch
    from tqdm import tqdm
    import time
    from opencood.hypes_yaml.yaml_utils import load_yaml
    test_data = torch.rand(3, 1, 4, 512, 512, 3)
    test_data = test_data.cuda()

    extrinsic = torch.rand(3, 1, 4, 4, 4)
    intrinsic = torch.rand(3, 1, 4, 3, 3)
    transformation_matrix = torch.from_numpy(np.identity(4))
    transformation_matrix = einops.repeat(transformation_matrix,
                                          'h w -> b l h w', b=1, l=3)
    record_len = torch.from_numpy(np.array([3], dtype=int))

    extrinsic = extrinsic.cuda()
    intrinsic = intrinsic.cuda()
    record_len = record_len.cuda()
    transformation_matrix = transformation_matrix.cuda()

    params = load_yaml('opencood/hypes_yaml/opcamera/corpbevt.yaml')
    params['model']['args']['max_cav'] = 3
    params['model']['args']['fax_fusion']['agent_size'] = 3
    model = PointPillarCrossViewTransformerFCooper(params['model']['args'])
    model = model.cuda()

    time_list = []
    batch = {'inputs': test_data,
             'extrinsic': extrinsic,
             'transformation_matrix': transformation_matrix,
             'record_len': record_len,
             'intrinsic': intrinsic}

    with torch.cuda.amp.autocast(enabled=False):
        with torch.no_grad():
            for _ in tqdm(range(1000 + 1)):
                start_time = time.time()
                model(batch)
                duration = time.time() - start_time
                time_list.append(duration)
