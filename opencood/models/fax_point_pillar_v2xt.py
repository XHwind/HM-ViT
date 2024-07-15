import torch
import torch.nn as nn

from opencood.models.fax_fused_transformer import FaxFusedTransformer
from opencood.models.point_pillar import PointPillar
from opencood.models.mwin_tranformer import V2XTransformer
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.spatial_transformation import \
    SpatialTransformation
from opencood.models.sub_modules.torch_transformation_utils import \
    get_roi_and_cav_mask
from opencood.models.base_camera_lidar_intermediate import \
    BaseCameraLiDARIntermediate
from opencood.models.sub_modules.naive_compress import NaiveCompressor


class FaxPointPillarV2XT(BaseCameraLiDARIntermediate):
    def __init__(self, config):
        super(FaxPointPillarV2XT, self).__init__(config)
        self.camera_encoder = FaxFusedTransformer(config['camera'])
        self.lidar_encoder = PointPillar(config['lidar'])

        self.compression = config['compression'] > 0
        if self.compression:
            self.compressor = NaiveCompressor(256, config['compression'])

        self.spatial_transform = SpatialTransformation(
            config['spatial_transform'])

        self.fusion_net = V2XTransformer(config['transformer'])

        self.set_return_features()

        self.cls_head = nn.Conv2d(256, config['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(256, 7 * config['anchor_number'],
                                  kernel_size=1)

    def forward(self, batch):
        # (B, L)
        mode = batch['mode']
        record_len = batch['record_len']

        max_cav = mode.shape[1]
        mode_unpack = self.unpad_mode_encoding(mode, record_len)

        camera_features = None
        lidar_features = None

        # If there is at least one camera
        if not torch.all(mode_unpack == 1):
            batch_camera = self.extract_camera_input(batch)
            camera_features = self.camera_encoder(batch_camera)
        # If there is at least one lidar
        if not torch.all(mode_unpack == 0):
            batch_lidar = self.extract_lidar_input(batch)
            lidar_features = self.lidar_encoder(batch_lidar)
        x = self.combine_features(camera_features,
                                  lidar_features, mode_unpack,
                                  record_len)
        if self.compression:
            x = self.compressor(x)

        # N, C, H, W -> B,  L, C, H, W
        x, mask = regroup(x, record_len, max_cav)

        transformation_matrix = batch['transformation_matrix']
        # B, L, C, H, W
        x = self.spatial_transform(x, transformation_matrix)
        # B, H, W, 1, L
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(
            3) if not self.use_roi_mask \
            else get_roi_and_cav_mask(
            (x.shape[0], x.shape[1], x.shape[3], x.shape[4], x.shape[2]),
            mask,
            transformation_matrix,
            self.discrete_ratio,
            self.downsample_rate)
        # B, C, H, W
        x = self.fusion_net(x.permute(0, 1, 3, 4, 2),
                            mode,
                            com_mask).squeeze(
            1).permute(0, 3, 1, 2)

        psm = self.cls_head(x)
        rm = self.reg_head(x)
        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict
