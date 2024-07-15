"""
Transform points to voxels using sparse conv library
"""
import sys

import numpy as np
import torch

from opencood.data_utils.pre_processor.base_preprocessor import \
    BasePreprocessor
from opencood.data_utils.pre_processor.voxel_preprocessor import \
    VoxelPreprocessor
from opencood.data_utils.pre_processor.bev_preprocessor import BevPreprocessor
from opencood.data_utils.pre_processor.sp_voxel_preprocessor import \
    SpVoxelPreprocessor
from opencood.data_utils.pre_processor.rgb_preprocessor import RgbPreProcessor

__all__ = {
    'BasePreprocessor': BasePreprocessor,
    'VoxelPreprocessor': VoxelPreprocessor,
    'BevPreprocessor': BevPreprocessor,
    'SpVoxelPreprocessor': SpVoxelPreprocessor,
    'RgbPreprocessor': RgbPreProcessor
}


class CamLiPreprocessor(BasePreprocessor):
    def __init__(self, preprocess_params, train):
        super(CamLiPreprocessor, self).__init__(preprocess_params,
                                                train)
        args = preprocess_params

        assert args['args']['camera_preprocess']['core_method'] in __all__ and \
               args['args']['lidar_preprocess']['core_method'] in __all__

        camera_preprocess_name = args['args']['camera_preprocess'][
            'core_method']
        camera_preprocess_cfg = args['args']['camera_preprocess']
        lidar_preprocess_name = args['args']['lidar_preprocess']['core_method']
        lidar_preprocess_cfg = args['args']['lidar_preprocess']

        self.lidar_preprocessor = __all__[lidar_preprocess_name](
            preprocess_params=lidar_preprocess_cfg,
            train=train
        )
        self.camera_preprocessor = __all__[camera_preprocess_name](
            preprocess_params=camera_preprocess_cfg,
            train=train
        )

    def preprocess(self, data, type='camera'):
        assert type in ['lidar',
                        'camera'], f"type must be either lidar or camera " \
                                   "but received {type}. "
        output = None

        if type == "lidar":
            output = self.lidar_preprocessor.preprocess(data)
        if type == "camera":
            output = self.camera_preprocessor.preprocess(data)

        return output

    def collate_batch(self, batch, type):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list or dict
            List or dictionary.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        assert type == "lidar", "only LiDAR data needs customized collate func"
        if isinstance(batch, list):
            return self.collate_batch_list(batch)
        elif isinstance(batch, dict):
            return self.collate_batch_dict(batch)
        else:
            sys.exit('Batch has too be a list or a dictionarn')

    def collate_batch_list(self, batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list
            List of dictionary. Each dictionary represent a single frame.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        return self.lidar_preprocessor.collate_batch(batch)

    @staticmethod
    def collate_batch_dict(batch: dict):
        """
        Collate batch if the batch is a dictionary,
        eg: {'voxel_features': [feature1, feature2...., feature n]}

        Parameters
        ----------
        batch : dict

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        voxel_features = \
            torch.from_numpy(np.concatenate(batch['voxel_features']))
        voxel_num_points = \
            torch.from_numpy(np.concatenate(batch['voxel_num_points']))
        coords = batch['voxel_coords']
        voxel_coords = []

        for i in range(len(coords)):
            voxel_coords.append(
                np.pad(coords[i], ((0, 0), (1, 0)),
                       mode='constant', constant_values=i))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {'voxel_features': voxel_features,
                'voxel_coords': voxel_coords,
                'voxel_num_points': voxel_num_points}
