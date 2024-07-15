import torch
import torch.nn as nn

from opencood.models.bevformer_wrapper import BEVFormerWrapper
from opencood.models.point_pillar import PointPillar
from opencood.models.fusion_modules.v2v_fuse import V2VNetFusion
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.spatial_transformation import \
    SpatialTransformation
from opencood.models.sub_modules.torch_transformation_utils import \
    get_roi_and_cav_mask
from opencood.models.base_camera_lidar_intermediate import \
    BaseCameraLiDARIntermediate
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.sub_modules.hetero_decoder import HeteroDecoder


class BevformerPointPillarV2VNet(BaseCameraLiDARIntermediate):
    def __init__(self, config):
        super(BevformerPointPillarV2VNet, self).__init__(config)
        self.camera_encoder = BEVFormerWrapper(config['camera'])
        self.lidar_encoder = PointPillar(config['lidar'])

        self.compression = config['compression'] > 0
        if self.compression:
            self.compressor = NaiveCompressor(256, config['compression'])

        self.spatial_transform = SpatialTransformation(
            config['spatial_transform'])

        self.fusion_net = V2VNetFusion(config['v2vfusion'])

        self.set_return_features()

        self.use_hetero_decoder = 'hetero_decoder' in config
        if 'decoder' in config:
            self.decoder = NaiveDecoder(config['decoder'])
        if 'hetero_decoder' in config:
            self.decoder = HeteroDecoder(config['hetero_decoder'])

        self.cls_head = nn.Conv2d(256, config['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(256, 7 * config['anchor_number'],
                                  kernel_size=1)
        self._fix_camera_backbone = False
        self._fix_lidar_backbone = False

    def fix_camera_backbone(self):
        self._fix_camera_backbone = True

    def fix_lidar_backbone(self):
        self._fix_lidar_backbone = True

    def _freeze_weights(self, model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    def pack(self, x, record_len):
        x_pack = []
        B = record_len.shape[0]
        for i in range(B):
            x_pack.append(x[i, :record_len[i]])
        return torch.cat(x_pack, dim=0)

    def forward(self, batch):
        if self._fix_lidar_backbone:
            self._freeze_weights(self.lidar_encoder)
        if self._fix_camera_backbone:
            self._freeze_weights((self.camera_encoder))
        # (B, L)
        mode = batch['mode']
        record_len = batch['record_len']
        pairwise_t_matrix = batch['pairwise_t_matrix']

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

        # N, C, H, W
        x = self.pack(x, record_len)
        # B, C, H, W
        x = self.fusion_net(x, mask, record_len, pairwise_t_matrix)

        if self.use_hetero_decoder:
            psm,rm = self.decoder(x.unsqueeze(1), mode, use_upsample=False)

        else:
            x = self.decoder(x.unsqueeze(1), use_upsample=False).squeeze(1)
            psm = self.cls_head(x)
            rm = self.reg_head(x)
        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict
