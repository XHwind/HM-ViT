import torch
import torch.nn as nn


from opencood.models.fax_fused_transformer import FaxFusedTransformer
from opencood.models.point_pillar import PointPillar
from opencood.models.base_transformer import BaseTransformer
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.spatial_transformation import \
    SpatialTransformation
from opencood.models.sub_modules.torch_transformation_utils import \
    get_roi_and_cav_mask
from opencood.models.base_camera_lidar_intermediate import BaseCameraLiDARIntermediate
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fusion_modules.v2v_fuse import V2VNetFusion

class FaxPointPillarV2VNet(BaseCameraLiDARIntermediate):
    def __init__(self, config):
        super(FaxPointPillarV2VNet, self).__init__(config)
        self.camera_encoder = FaxFusedTransformer(config['camera'])
        self.lidar_encoder = PointPillar(config['lidar'])

        self.compression = config['compression'] > 0
        if self.compression:
            self.compressor = NaiveCompressor(256, config['compression'])

        self.spatial_transform = SpatialTransformation(
            config['spatial_transform'])


        self.fusion_net = V2VNetFusion(config['v2vnet_fusion'])

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

        pairwise_t_matrix = batch['pairwise_t_matrix']

        # B, C, H, W
        x = self.fusion_net(self.unpad_features(x, record_len),
                            record_len,
                            pairwise_t_matrix,
                            None).squeeze(1)

        psm = self.cls_head(x)
        rm = self.reg_head(x)
        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict
