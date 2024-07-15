import torch
from torch import nn
from mmdet3d.models import build_detector

from opencood.models.mmdet3d_plugin import *
from opencood.models.sub_modules.naive_decoder import NaiveDecoder


class BEVFormerWrapper(nn.Module):
    def __init__(self, config):
        super(BEVFormerWrapper, self).__init__()
        cfg = config['BEVFormer']['cfg']
        self.cfg = cfg
        self.config = config
        self.bevformer = build_detector(cfg.model)
        self.img_shape = config['BEVFormer']['img_shape']

        # decoder params
        decoder_params = config['decoder']

        self.decoder = NaiveDecoder(decoder_params)

        self.cls_head = nn.Conv2d(256, config['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(256, 7 * config['anchor_number'],
                                  kernel_size=1)
        self.return_features = False

    def reformat_input(self, batch):
        # (B,N,H,W,3)
        img = batch['camera'].permute(0, 1, 4, 2, 3)
        bs = img.shape[0]
        num_cam = img.shape[1]

        # cam2ego = batch['extrinsic']
        # ego2cam = torch.linalg.inv(cam2ego)
        cav2cam = batch['cav2cam_extrinsic']

        intrinsics = batch['intrinsic']
        intrinsics_hom = torch.eye(4).reshape((1, 1, 4, 4)).repeat(
            (bs, num_cam, 1, 1)).to(intrinsics.device)
        intrinsics_hom[:, :, :3, :3] = intrinsics
        # New we must change from UE4's coordinate system to an "standard"
        # camera coordinate system (the same used by OpenCV):
        # ^ z                       . z
        # |                        /
        # |              to:      +-------> x
        # | . x                   |
        # |/                      |
        # +-------> y             v y
        # (x, y ,z) -> (y, -z, x)
        flip_matrix = torch.Tensor(
            [[0, 1, 0, 0],
             [0, 0, -1, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 1]]).reshape(
            (1, 1, 4, 4)).repeat(
            (bs, num_cam, 1, 1)).to(intrinsics.device)
        flip_matrix[..., 1, 1] = -1

        lidar2img = torch.matmul(flip_matrix, cav2cam)
        lidar2img = torch.matmul(intrinsics_hom,
                                 lidar2img).detach().cpu().numpy()
        img_metas = []
        for i in range(bs):
            img_meta = {}
            img_meta['img_shape'] = [(self.img_shape[0], self.img_shape[1]) for
                                     _ in range(num_cam)]
            img_meta['lidar2img'] = []
            for j in range(num_cam):
                img_meta['lidar2img'].append(lidar2img[i, j])
            img_metas.append(img_meta)
        return img, img_metas
    def set_return_features(self):
        self.return_features = True

    def forward(self, batch):
        img, img_metas = self.reformat_input(batch)
        bev = self.bevformer.forward_train(img=img, img_metas=img_metas,
                                           only_bev=True)
        # (bs,bh,bw,C)
        bev = bev.unflatten(1, [self.cfg.bev_h_, self.cfg.bev_w_])
        # (B, C, H, W)
        bev = bev.permute(0, 3, 1, 2)
        if self.return_features:
            return bev
        bev = bev.unsqueeze(1)
        # (bs,1,C,H,W)
        x = self.decoder(bev, use_upsample=False).squeeze(1)
        psm = self.cls_head(x)
        rm = self.reg_head(x)
        output_dict = {'psm': psm,
                       'rm': rm}
        return output_dict
