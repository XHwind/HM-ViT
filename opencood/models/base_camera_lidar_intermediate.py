import torch
import torch.nn as nn

class BaseCameraLiDARIntermediate(nn.Module):
    def __init__(self, config):
        super(BaseCameraLiDARIntermediate, self).__init__()
        self.camera_encoder = None
        self.lidar_encoder = None
        self.fusion_net = None

        self.downsample_rate = config['spatial_transform']['downsample_rate']
        self.discrete_ratio = config['spatial_transform']['voxel_size'][0]
        self.use_roi_mask = config['spatial_transform']['use_roi_mask']

    def set_return_features(self):
        self.camera_encoder.set_return_features()
        self.lidar_encoder.set_return_features()

    def extract_camera_input(self, batch):
        mode = batch['mode']
        record_len = batch['record_len']
        mode = self.unpad_mode_encoding(mode, record_len).to(torch.int)
        batch_camera = {
            'camera': batch['camera'][mode == 0, ...],
            'intrinsic': batch['intrinsic'][mode == 0, ...],
            'extrinsic': batch['extrinsic'][mode == 0, ...],
            'cav2cam_extrinsic': batch['cav2cam_extrinsic'][mode == 0, ...]
        }
        return batch_camera

    def extract_lidar_input(self, batch):
        mode = batch['mode']
        record_len = batch['record_len']
        mode = self.unpad_mode_encoding(mode, record_len).to(torch.int)

        processed_lidar_all = batch['processed_lidar']
        voxel_features_list = []
        voxel_coords_list = []
        voxel_num_points_list = []

        count = 0
        for i in range(len(mode)):
            if mode[i] != 1:
                continue
            coords = processed_lidar_all['voxel_coords']
            mask = coords[:, 0] == i

            coords[mask, 0] = count

            voxel_coords_list.append(coords[mask, :])
            voxel_features_list.append(
                processed_lidar_all['voxel_features'][mask, :])
            voxel_num_points_list.append(
                processed_lidar_all['voxel_num_points'][mask])
            count += 1
        processed_lidar_selected = {
            'voxel_features': torch.cat(voxel_features_list, dim=0),
            'voxel_coords': torch.cat(voxel_coords_list, dim=0),
            'voxel_num_points': torch.cat(voxel_num_points_list, dim=0)

        }
        batch_lidar = {
            'processed_lidar': processed_lidar_selected
        }
        return batch_lidar

    def unpad_mode_encoding(self, mode, record_len):
        B = mode.shape[0]
        out = []
        for i in range(B):
            out.append(mode[i, :record_len[i]])
        return torch.cat(out, dim=0)
    def unpad_features(self, x, record_len):
        B = x.shape[0]
        out = []
        for i in range(B):
            out.append(x[i, :record_len[i],...])
        return torch.cat(out, dim=0)

    def combine_features(self, camera_feature, lidar_feature, mode,
                         record_len):
        combined_features = []
        if len(mode.shape) == 2:
            mode = self.unpad_mode_encoding(mode, record_len)
        camera_count = 0
        lidar_count = 0
        for i in range(len(mode)):
            if mode[i] == 0:
                combined_features.append(camera_feature[camera_count, ...])
                camera_count += 1
            elif mode[i] == 1:
                combined_features.append(lidar_feature[lidar_count, ...])
                lidar_count += 1
            else:
                raise ValueError(f"Mode but be either 1 or 0 but received "
                                 f"{mode[i]}")
        # print(f"Lidar/camera = {lidar_count} / {camera_count}")
        combined_features = torch.stack(combined_features, dim=0)
        return combined_features

    def forward(self, batch):
        return batch