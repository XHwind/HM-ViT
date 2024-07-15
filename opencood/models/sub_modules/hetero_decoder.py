import torch
from torch import nn

from opencood.models.sub_modules.naive_decoder import NaiveDecoder


class HeteroDecoder(nn.Module):
    """
    A Naive decoder implementation

    Parameters
    ----------
    params: dict

    Attributes
    ----------
    num_ch_dec : list
        The decoder layer channel numbers.

    num_layer : int
        The number of decoder layers.

    input_dim : int
        The channel number of the input to
    """

    def __init__(self, params):
        super(HeteroDecoder, self).__init__()
        input_dim = params['num_ch_dec'][0]
        self.camera_decoder = NaiveDecoder(params)
        self.lidar_decoder = NaiveDecoder(params)

        self.camera_cls_head = nn.Conv2d(input_dim, params['anchor_number'],
                                  kernel_size=1)
        self.camera_reg_head = nn.Conv2d(input_dim, 7 * params['anchor_number'],
                                  kernel_size=1)
        self.lidar_cls_head = nn.Conv2d(input_dim, params['anchor_number'],
                                  kernel_size=1)
        self.lidar_reg_head = nn.Conv2d(input_dim, 7 * params['anchor_number'],
                                  kernel_size=1)

    def forward(self, x, mode, use_upsample=True):
        """
        Upsample to

        Parameters
        ----------
        x : torch.tensor
            The bev bottleneck feature, shape: (B, L, C1, H, W)

        Returns
        -------
        Output features with (B, L, C2, H, W)
        """
        ego_mode = mode[:, 0]
        camera_psm, camera_rm, lidar_psm, lidar_rm = None, None, None, None

        # If there is at least one camera
        if not torch.all(ego_mode == 1):
            camera_feature = x[ego_mode == 0, ...]
            camera_feature = self.camera_decoder(camera_feature, use_upsample=use_upsample).squeeze(1)
            camera_psm = self.camera_cls_head(camera_feature)
            camera_rm = self.camera_reg_head(camera_feature)
        # If there is at least one lidar
        if not torch.all(ego_mode == 0):
            lidar_feature = x[ego_mode == 1, ...]
            lidar_feature = self.lidar_decoder(lidar_feature, use_upsample=use_upsample).squeeze(1)
            lidar_psm = self.lidar_cls_head(lidar_feature)
            lidar_rm = self.lidar_reg_head(lidar_feature)

        psm = self.combine_features(camera_psm, lidar_psm, ego_mode)
        rm = self.combine_features(camera_rm, lidar_rm, ego_mode)

        return psm, rm
    def combine_features(self, camera, lidar, ego_mode):
        combined_features = []
        camera_count = 0
        lidar_count = 0
        for i in range(len(ego_mode)):
            if ego_mode[i] == 0:
                combined_features.append(camera[camera_count, ...])
                camera_count += 1
            elif ego_mode[i] == 1:
                combined_features.append(lidar[lidar_count, ...])
                lidar_count += 1
            else:
                raise ValueError(f"Mode but be either 1 or 0 but received "
                                 f"{ego_mode[i]}")
        combined_features = torch.stack(combined_features, dim=0)
        return combined_features