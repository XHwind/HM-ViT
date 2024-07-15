from opencood.data_utils.datasets.camera_only.base_camera_dataset import BaseCameraDataset
from opencood.data_utils.datasets.lidar_only.late_fusion_dataset import LateFusionDataset
from opencood.data_utils.datasets.lidar_only.early_fusion_dataset import EarlyFusionDataset
from opencood.data_utils.datasets.lidar_only.intermediate_fusion_dataset import IntermediateFusionDataset
from opencood.data_utils.datasets.camera_only.late_fusion_dataset import CamLateFusionDataset
from opencood.data_utils.datasets.camera_only.intermediate_fusion_dataset import CamIntermediateFusionDataset
from opencood.data_utils.datasets.mixed.base_camera_lidar_dataset import BaseCameraLiDARDataset
from opencood.data_utils.datasets.mixed.intermediate_fusion_dataset import CamLiIntermediateFusionDataset
from opencood.data_utils.datasets.mixed.late_fusion_dataset import CamLiLateFusionDataset
__all__ = {
    'LateFusionDataset': LateFusionDataset,
    'EarlyFusionDataset': EarlyFusionDataset,
    'IntermediateFusionDataset': IntermediateFusionDataset,
    'BaseCameraDataset': BaseCameraDataset,
    'CamLateFusionDataset': CamLateFusionDataset,
    'CamIntermediateFusionDataset': CamIntermediateFusionDataset,
    'BaseCameraLiDARDataset': BaseCameraLiDARDataset,
    'CamLiIntermediateFusionDataset': CamLiIntermediateFusionDataset,
    'CamLiLateFusionDataset': CamLiLateFusionDataset
}

# the final range for evaluation
# GT_RANGE = [-102.4, -102.4, -3, 102.4, 102.4, 1]# [-140, -40, -3, 140, 40, 1]
GT_RANGE = [-102.4, -102.4, -3, 102.4, 102.4, 1]
CAMERA_GT_RANGE = [-50, -50, -3, 50, 50, 1]
# The communication range for cavs
COM_RANGE = 50


def build_dataset(dataset_cfg, visualize=False, train=True, validate=False):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"
    assert dataset_name in __all__.keys(), error_message

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train,
        validate=validate
    )

    return dataset
