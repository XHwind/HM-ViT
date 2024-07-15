"""
Fusion for intermediate level (camera)
"""
from collections import OrderedDict

import numpy as np
import torch

import opencood
from opencood.data_utils.datasets.mixed import base_camera_lidar_dataset
from opencood.utils import common_utils
from opencood.utils.pcd_utils import downsample_lidar_minimum


class CamLiIntermediateFusionDataset(
    base_camera_lidar_dataset.BaseCameraLiDARDataset):
    def __init__(self, params, visualize, train=True, validate=False):
        super(CamLiIntermediateFusionDataset, self).__init__(params,
                                                             visualize,
                                                             train,
                                                             validate)
        self.visible = params['train_params']['visible']

    def __getitem__(self, idx):
        data_sample = self.get_sample_random(idx)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = OrderedDict()

        ego_id = -999
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in data_sample.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break
        assert cav_id == list(data_sample.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -999
        assert len(ego_lidar_pose) > 0

        pairwise_t_matrix = \
            self.get_pairwise_transformation(data_sample,
                                             self.params['train_params'][
                                                 'max_cav'])

        # Final shape: (L, M, H, W, 3)
        camera_data = []
        origin_camera = []
        # (L, M, 3, 3)
        camera_intrinsic = []
        # (L, M, 4, 4)
        camera2ego = []
        cav2cam = []

        # (max_cav, 4, 4)
        transformation_matrix = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in data_sample.items():
            distance = common_utils.cav_distance_cal(selected_cav_base,
                                                     ego_lidar_pose)
            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue

            selected_cav_processed = \
                self.get_single_cav(selected_cav_base)

            camera_data.append(selected_cav_processed['camera']['data'])
            origin_camera.append(
                selected_cav_processed['camera']['origin_data'])
            camera_intrinsic.append(
                selected_cav_processed['camera']['intrinsic'])
            camera2ego.append(
                selected_cav_processed['camera']['extrinsic'])
            cav2cam.append(selected_cav_processed['camera']['cav2cam'])
            transformation_matrix.append(
                selected_cav_processed['transformation_matrix'])

        # stack all agents together
        camera_data = np.stack(camera_data)
        origin_camera = np.stack(origin_camera)
        camera_intrinsic = np.stack(camera_intrinsic)
        camera2ego = np.stack(camera2ego)
        cav2cam = np.stack(cav2cam)

        # gt_dynamic = np.stack(gt_dynamic)
        # gt_static = np.stack(gt_static)

        # padding
        transformation_matrix = np.stack(transformation_matrix)
        padding_eye = np.tile(np.eye(4)[None], (self.max_cav - len(
            transformation_matrix), 1, 1))
        transformation_matrix = np.concatenate(
            [transformation_matrix, padding_eye], axis=0)

        processed_lidar_features = data_sample[ego_id][
            'processed_lidar_features']
        merged_feature_dict = self.merge_features_to_dict(
            processed_lidar_features)

        processed_data_dict['ego'].update({
            'transformation_matrix': transformation_matrix,
            'pairwise_t_matrix': pairwise_t_matrix,
            'camera_data': camera_data,
            'camera_intrinsic': camera_intrinsic,
            'camera_extrinsic': camera2ego,
            'cav2cam_extrinsic': cav2cam,
            # detection related
            'object_bbx_center': data_sample[ego_id]['object_bbx_ego'],
            'object_bbx_mask': data_sample[ego_id]['object_bbx_mask'],
            'object_ids': data_sample[ego_id]['object_ids'],
            'anchor_box': data_sample[ego_id]['anchor_box'],
            'processed_lidar': merged_feature_dict,
            'label_dict': data_sample[ego_id]['label_dict'],
            'cav_num': data_sample[ego_id]['cav_num'],
            'velocity': data_sample[ego_id]['velocity'],
            'time_delay': data_sample[ego_id]['time_delay'],
            'infra': data_sample[ego_id]['infra'],
            'mode': data_sample[ego_id]['mode']
        })
        if self.visualize:
            processed_data_dict['ego'].update({
                'origin_lidar': data_sample[ego_id]['projected_lidar'],
                'origin_camera': origin_camera
            })
        return processed_data_dict

    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature)

        return merged_feature_dict

    @staticmethod
    def get_pairwise_transformation(base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4)
        """
        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))
        # default are identity matrix
        pairwise_t_matrix[:, :] = np.identity(4)

        # return pairwise_t_matrix

        t_list = []

        # save all transformation matrix in a list in order first.
        for cav_id, cav_content in base_data_dict.items():
            t_list.append(cav_content['params']['transformation_matrix'])

        for i in range(len(t_list)):
            for j in range(len(t_list)):
                # identity matrix to self
                if i == j:
                    continue
                # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix

    def get_single_cav(self, selected_cav_base):
        """
        Process the cav data in a structured manner for intermediate fusion.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = OrderedDict()

        # update the transformation matrix
        transformation_matrix = \
            selected_cav_base['params']['transformation_matrix']
        selected_cav_processed.update({
            'transformation_matrix': transformation_matrix
        })

        # for intermediate fusion, we only need ego's gt
        if selected_cav_base['ego']:
            # process the groundtruth
            # selected_cav_processed.update({'gt': gt_dict})
            pass

        all_camera_data = []
        all_origin_camera = []
        all_camera_intrinsic = []
        # camera to ego vehicle not cav!
        all_camera_extrinsic = []
        # cav lidar to camera
        all_cav2cam = []

        # preprocess the input rgb image and extrinsic params first
        for camera_id, camera_data in selected_cav_base['camera_np'].items():
            all_origin_camera.append(camera_data)
            camera_data = self.pre_processor.preprocess(camera_data,
                                                        type='camera')
            camera_intrinsic = \
                selected_cav_base['camera_params'][camera_id][
                    'camera_intrinsic']
            cam2ego = \
                selected_cav_base['camera_params'][camera_id][
                    'camera_extrinsic_to_ego']
            cav2cam = selected_cav_base['camera_params'][camera_id][
                'camera_extrinsic']

            all_camera_data.append(camera_data)
            all_camera_intrinsic.append(camera_intrinsic)
            all_camera_extrinsic.append(cam2ego)
            all_cav2cam.append(cav2cam)

        camera_dict = {
            'origin_data': np.stack(all_origin_camera),
            'data': np.stack(all_camera_data),
            'intrinsic': np.stack(all_camera_intrinsic),
            'extrinsic': np.stack(all_camera_extrinsic),
            'cav2cam': np.stack(all_cav2cam)
        }

        selected_cav_processed.update({'camera': camera_dict})

        return selected_cav_processed

    def collate_batch(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        if not self.train:
            assert len(batch) == 1

        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []
        processed_lidar_list = []

        cam_rgb_all_batch = []
        cam_to_ego_all_batch = []
        cav2cam_all_batch = []
        cam_intrinsic_all_batch = []

        transformation_matrix_all_batch = []
        pairwise_t_matrix_all_batch = []
        # used to save each scenario's agent number
        record_len = []
        label_dict_list = []

        # used for PriorEncoding
        velocity = []
        time_delay = []
        infra = []
        mode = []

        if self.visualize:
            projected_lidar = []
            origin_camera = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']

            camera_data = ego_dict['camera_data']
            camera_intrinsic = ego_dict['camera_intrinsic']
            camera_extrinsic = ego_dict['camera_extrinsic']
            cav2cam = ego_dict['cav2cam_extrinsic']

            cam_rgb_all_batch.append(camera_data)
            cam_intrinsic_all_batch.append(camera_intrinsic)
            cam_to_ego_all_batch.append(camera_extrinsic)
            cav2cam_all_batch.append(cav2cam)

            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            object_ids.append(ego_dict['object_ids'])

            processed_lidar_list.append(ego_dict['processed_lidar'])
            record_len.append(ego_dict['cav_num'])
            label_dict_list.append(ego_dict['label_dict'])

            velocity.append(ego_dict['velocity'])
            time_delay.append(ego_dict['time_delay'])
            infra.append(ego_dict['infra'])
            mode.append(ego_dict['mode'])

            assert camera_data.shape[0] == \
                   camera_intrinsic.shape[0] == \
                   camera_extrinsic.shape[0]
            assert camera_data.shape[0] == ego_dict['cav_num']

            # transformation matrix
            transformation_matrix_all_batch.append(
                ego_dict['transformation_matrix'])
            # pairwise matrix
            pairwise_t_matrix_all_batch.append(ego_dict['pairwise_t_matrix'])

            if self.visualize:
                projected_lidar.append(ego_dict['origin_lidar'])
                origin_camera.append(ego_dict['origin_camera'])

        # (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(merged_feature_dict, type='lidar')

        # (B*L, M, H, W, C)
        # L: number of agents; M: number of camera per vehicle
        cam_rgb_all_batch = torch.from_numpy(
            np.concatenate(cam_rgb_all_batch, axis=0)).float()
        cam_intrinsic_all_batch = torch.from_numpy(
            np.concatenate(cam_intrinsic_all_batch, axis=0)).float()
        cam_to_ego_all_batch = torch.from_numpy(
            np.concatenate(cam_to_ego_all_batch, axis=0)).float()
        cav2cam_all_batch = torch.from_numpy(
            np.concatenate(cav2cam_all_batch, axis=0)).float()
        # (B,)
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)

        # (B, max_cav)
        velocity = torch.from_numpy(np.array(velocity))
        time_delay = torch.from_numpy(np.array(time_delay))
        infra = torch.from_numpy(np.array(infra))
        mode = torch.from_numpy(np.array(mode))
        # (B, max_cav, 3)
        prior_encoding = \
            torch.stack([velocity, time_delay, infra], dim=-1).float()

        # (B,max_cav,4,4)
        transformation_matrix_all_batch = \
            torch.from_numpy(np.stack(transformation_matrix_all_batch)).float()
        pairwise_t_matrix_all_batch = \
            torch.from_numpy(np.stack(pairwise_t_matrix_all_batch)).float()

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict['ego'].update({
            'object_bbx_center': object_bbx_center,
            'object_bbx_mask': object_bbx_mask,
            'label_dict': label_torch_dict,
            'object_ids': object_ids[0],
            # inputs
            'camera': cam_rgb_all_batch,
            'processed_lidar': processed_lidar_torch_dict,
            # intrinsic/extrinsic
            'extrinsic': cam_to_ego_all_batch,
            'cav2cam_extrinsic': cav2cam_all_batch,
            'intrinsic': cam_intrinsic_all_batch,
            'transformation_matrix': transformation_matrix_all_batch,
            'pairwise_t_matrix': pairwise_t_matrix_all_batch,
            'record_len': record_len,
            'prior_encoding': prior_encoding,
            'mode': mode
        })

        if self.visualize:
            projected_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=projected_lidar))
            projected_lidar = torch.from_numpy(projected_lidar)
            origin_camera = torch.from_numpy(
                np.concatenate(origin_camera, axis=0))
            output_dict['ego'].update({'origin_lidar': projected_lidar,
                                       'origin_camera': origin_camera})
        if not self.train:
            # check if anchor box in the batch
            if batch[0]['ego']['anchor_box'] is not None:
                output_dict['ego'].update({'anchor_box':
                    torch.from_numpy(np.array(
                        batch[0]['ego'][
                            'anchor_box']))})
            output_dict['ego'].update({'no_post_projection': True})
            # save the transformation matrix (4, 4) to ego vehicle
            # transformation_matrix_torch = \
            #     torch.from_numpy(np.identity(4)).float()
            # output_dict['ego'].update({'transformation_matrix':
            #                                transformation_matrix_torch})
        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor
