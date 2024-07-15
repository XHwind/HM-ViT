"""
Late fusion for camera.
"""

import random
from collections import OrderedDict

import numpy as np
import torch

import opencood
from opencood.data_utils.datasets.mixed import base_camera_lidar_dataset
from opencood.utils import common_utils
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils import box_utils


class CamLiLateFusionDataset(base_camera_lidar_dataset.BaseCameraLiDARDataset):
    def __init__(self, params, visualize, train=True, validate=False):
        super(CamLiLateFusionDataset, self).__init__(params, visualize, train,
                                                     validate)
        self.visible = params['train_params']['visible']

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx, True)
        if self.train:
            return self.get_item_train(base_data_dict)
        else:
            return self.get_item_test(base_data_dict)

    def get_item_train(self, base_data_dict):
        processed_data_dict = OrderedDict()

        # during training, we return a random cav's data
        if not self.visualize:
            selected_cav_id, selected_cav_base = \
                random.choice(list(base_data_dict.items()))
        else:
            selected_cav_id, selected_cav_base = \
                list(base_data_dict.items())[0]

        selected_cav_processed = \
            self.get_single_cav(selected_cav_base)

        processed_data_dict.update({'ego': selected_cav_processed})

        return processed_data_dict

    def get_item_test(self, base_data_dict):
        processed_data_dict = OrderedDict()
        ego_id = -999
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break

        assert cav_id == list(base_data_dict.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -999
        assert len(ego_lidar_pose) > 0

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            distance = common_utils.cav_distance_cal(selected_cav_base,
                                                     ego_lidar_pose)
            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue

            selected_cav_processed = \
                self.get_single_cav(selected_cav_base)
            selected_cav_processed['mode'] = selected_cav_base['mode']

            update_cav = "ego" if cav_id == ego_id else cav_id
            processed_data_dict.update({update_cav: selected_cav_processed})

        return processed_data_dict

    def get_single_cav(self, selected_cav_base):
        """
        Process the cav data in a structured manner for late fusion.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = OrderedDict({'camera': OrderedDict()})
        transformation_matrix = \
            selected_cav_base['params']['transformation_matrix']
        selected_cav_processed.update({'transformation_matrix':
                                           transformation_matrix})
        all_origin_camera = []
        # preprocess the input rgb image and extrinsic params first
        for camera_id, camera_data in selected_cav_base['camera_np'].items():
            all_origin_camera.append(camera_data)
            camera_data = self.pre_processor.preprocess(camera_data)

            camera_intrinsic = \
                selected_cav_base['camera_params'][camera_id][
                    'camera_intrinsic']
            cam2ego = \
                selected_cav_base['camera_params'][camera_id][
                    'camera_extrinsic_to_ego']
            cav2cam = selected_cav_base['camera_params'][camera_id][
                'camera_extrinsic']

            camera_dict = {
                'data': camera_data,
                'intrinsic': camera_intrinsic,
                'extrinsic': cam2ego,
                'cav2cam': cav2cam
            }
            camera_dict.update({'origin_camera': all_origin_camera})

            selected_cav_processed['camera'].update({camera_id:
                                                         camera_dict})
        # Process the LiDAR data
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
        # remove points that hit ego vehicle
        lidar_np = mask_ego_points(lidar_np)

        # generate the bounding box(n, 7) under the cav's space
        object_bbx_center, object_bbx_mask, object_ids = \
            self.post_processor.generate_object_center([selected_cav_base],
                                                       selected_cav_base[
                                                           'params'][
                                                           'lidar_pose'])
        # data augmentation
        lidar_np, object_bbx_center, object_bbx_mask = \
            self.augment(lidar_np, object_bbx_center, object_bbx_mask)

        if self.visualize:
            selected_cav_processed.update({'origin_lidar': lidar_np})

        lidar_dict = self.pre_processor.preprocess(lidar_np, type='lidar')
        selected_cav_processed.update({'processed_lidar': lidar_dict})

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()
        selected_cav_processed.update({'anchor_box': anchor_box})

        selected_cav_processed.update({'object_bbx_center': object_bbx_center,
                                       'object_bbx_mask': object_bbx_mask,
                                       'object_ids': object_ids})

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=object_bbx_mask)
        selected_cav_processed.update({'label_dict': label_dict})

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
        if self.train:
            return self.collate_batch_train(batch)
        else:
            return self.collate_batch_test(batch)

    def collate_batch_train(self, batch):
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
        # For training, the agent dimension is 1
        assert all(len(batch[i]) == 1 for i in range(len(batch)))

        output_dict = {'ego': {}}
        # collect gt
        object_bbx_center_all_batch = []
        object_bbx_mask_all_batch = []

        # collect lidar related data
        processed_lidar_all_batch = []
        label_dict_all_batch = []

        # collect camera related data
        cam_rgb_all_batch = []
        origin_cam_all_batch = []

        cam_to_ego_all_batch = []
        cav2cam_all_batch = []
        cam_intrinsic_all_batch = []

        transformation_matrix_all_batch = []

        origin_lidar_all_batch = []

        # loop all scenes
        for i in range(len(batch)):
            cur_scene_data = batch[i]

            # gt
            object_bbx_center_all_batch.append(
                cur_scene_data['ego']['object_bbx_center'])
            object_bbx_mask_all_batch.append(
                cur_scene_data['ego']['object_bbx_mask'])
            # lidar data
            processed_lidar_all_batch.append(
                cur_scene_data['ego']['processed_lidar'])
            label_dict_all_batch.append(
                cur_scene_data['ego']['label_dict'])
            if self.visualize:
                origin_lidar_all_batch.append(
                    cur_scene_data['ego']['origin_lidar'])

            # camera data
            camera_data = cur_scene_data['ego']['camera']

            cam_rgb_cur_agent = []
            origin_cam_cur_agent = []
            cam_to_ego_cur_agent = []
            cav2cam_cur_agent = []
            cam_intrinsic_cur_agent = []

            # loop all cameras
            for camera_id, camera_content in camera_data.items():
                cam_rgb_cur_agent.append(camera_content['data'])
                origin_cam_cur_agent.append(camera_content['origin_camera'])
                cam_to_ego_cur_agent.append(camera_content['extrinsic'])
                cav2cam_cur_agent.append(camera_content['cav2cam'])
                cam_intrinsic_cur_agent.append(camera_content['intrinsic'])

            # M, H, W, 3 -> M is the num of cameras
            cam_rgb_cur_agent = np.stack(cam_rgb_cur_agent)
            origin_cam_cur_agent = np.stack(origin_cam_cur_agent)
            # M, 4, 4
            cam_to_ego_cur_agent = np.stack(cam_to_ego_cur_agent)
            cav2cam_cur_agent = np.stack(cav2cam_cur_agent)
            # M, 3, 3
            cam_intrinsic_cur_agent = np.stack(cam_intrinsic_cur_agent)

            cam_rgb_all_batch.append(cam_rgb_cur_agent)
            origin_cam_all_batch.append(origin_cam_cur_agent)
            cam_to_ego_all_batch.append(cam_to_ego_cur_agent)
            cav2cam_all_batch.append(cav2cam_cur_agent)
            cam_intrinsic_all_batch.append(cam_intrinsic_cur_agent)

            transformation_matrix = \
                cur_scene_data['ego']['transformation_matrix']
            transformation_matrix_all_batch.append(transformation_matrix)

        # groundtruth gather
        # (B, max_num,7) -> max_num=100
        object_bbx_center_all_batch = \
            torch.from_numpy(np.stack(object_bbx_center_all_batch))
        # (B, max_num)
        object_bbx_mask_all_batch = \
            torch.from_numpy(np.stack(object_bbx_mask_all_batch))

        # lidar data gather
        processed_lidar_all_batch = self.pre_processor.collate_batch(
            processed_lidar_all_batch, type='lidar')

        label_dict_all_batch = self.post_processor.collate_batch(
            label_dict_all_batch)

        # camera data gather
        # (B,M,H,W,C)
        cam_rgb_all_batch = \
            torch.from_numpy(np.stack(cam_rgb_all_batch)).float()
        if self.visualize:
            origin_cam_all_batch = torch.from_numpy(
                np.stack(origin_cam_all_batch)).float()
        cam_to_ego_all_batch = \
            torch.from_numpy(np.stack(cam_to_ego_all_batch)).float()
        cav2cam_all_batch = torch.from_numpy(
            np.stack(cav2cam_all_batch)).float()
        cam_intrinsic_all_batch = \
            torch.from_numpy(np.stack(cam_intrinsic_all_batch)).float()
        # (B,4,4)
        transformation_matrix_all_batch = \
            torch.from_numpy(np.stack(transformation_matrix_all_batch)).float()

        output_dict['ego'].update({
            'object_bbx_center': object_bbx_center_all_batch,
            'object_bbx_mask': object_bbx_mask_all_batch,
            'label_dict': label_dict_all_batch,
            # inputs
            'camera': cam_rgb_all_batch,
            'processed_lidar': processed_lidar_all_batch,
            # intrinsic/extrinsic
            'extrinsic': cam_to_ego_all_batch,
            'cav2cam_extrinsic': cav2cam_all_batch,
            'intrinsic': cam_intrinsic_all_batch,
            'transformation_matrix': transformation_matrix_all_batch
        })
        if self.visualize:
            output_dict['ego'].update({
                'origin_camera': origin_cam_all_batch
            })

        return output_dict

    def collate_batch_test(self, batch):
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
        assert len(batch) == 1, "Batch size 1 is required during testing!"
        batch = batch[0]

        output_dict = {}
        mode = []
        if self.visualize:
            projected_lidar_list = []
            origin_lidar = []

        # loop all agents
        for cav_id, cav_content in batch.items():
            output_dict.update({cav_id: {}})
            mode.append(1 if cav_content['mode'] == 'lidar' else 0)
            # gt
            object_bbx_center = \
                torch.from_numpy(np.array([cav_content['object_bbx_center']]))
            object_bbx_mask = \
                torch.from_numpy(np.array([cav_content['object_bbx_mask']]))
            object_ids = cav_content['object_ids']

            # the anchor box is the same for all bounding boxes usually, thus
            # we don't need the batch dimension.
            if cav_content['anchor_box'] is not None:
                output_dict[cav_id].update({'anchor_box':
                    torch.from_numpy(np.array(
                        cav_content[
                            'anchor_box']))})

            # lidar data
            processed_lidar_torch_dict = \
                self.pre_processor.collate_batch(
                    [cav_content['processed_lidar']], type='lidar')
            # label dictionary
            label_torch_dict = \
                self.post_processor.collate_batch([cav_content['label_dict']])

            # camera data
            camera_data = cav_content['camera']

            cam_rgb_cur_agent = []
            origin_cam_cur_agent = []
            cam_to_ego_cur_agent = []
            cav2cam_cur_agent = []
            cam_intrinsic_cur_agent = []

            # loop all cameras
            for camera_id, camera_content in camera_data.items():
                cam_rgb_cur_agent.append(camera_content['data'])
                origin_cam_cur_agent.append(camera_content['origin_camera'])
                cam_to_ego_cur_agent.append(camera_content['extrinsic'])
                cav2cam_cur_agent.append(camera_content['cav2cam'])
                cam_intrinsic_cur_agent.append(camera_content['intrinsic'])

            # 1, M, H, W, 3 -> M is the num of cameras
            cam_rgb_cur_agent = torch.from_numpy(
                np.stack(cam_rgb_cur_agent)).float().unsqueeze(0)
            # 1, M, 4, 4
            cam_to_ego_cur_agent = torch.from_numpy(
                np.stack(cam_to_ego_cur_agent)).float().unsqueeze(0)
            cav2cam_cur_agent = torch.from_numpy(
                np.stack(cav2cam_cur_agent)).float().unsqueeze(0)
            # 1, M, 3, 3
            cam_intrinsic_cur_agent = torch.from_numpy(
                np.stack(cam_intrinsic_cur_agent)).float().unsqueeze(0)

            # save the transformation matrix (4, 4) to ego vehicle
            transformation_matrix = cav_content['transformation_matrix']
            transformation_matrix_torch = \
                torch.from_numpy(np.array(transformation_matrix)).float()

            output_dict[cav_id].update({
                'object_bbx_center': object_bbx_center,
                'object_bbx_mask': object_bbx_mask,
                'object_ids': object_ids,
                'label_dict': label_torch_dict,
                # inputs
                'camera': cam_rgb_cur_agent,
                'processed_lidar': processed_lidar_torch_dict,
                # intrinsic/extrinsic
                'extrinsic': cam_to_ego_cur_agent,
                'cav2cam_extrinsic': cav2cam_cur_agent,
                'intrinsic': cam_intrinsic_cur_agent,
                'transformation_matrix': transformation_matrix_torch,
                'mode': cav_content['mode']
            })
            if self.visualize:
                origin_lidar = \
                    np.array(downsample_lidar_minimum(
                        pcd_np_list=[cav_content['origin_lidar']]))
                origin_lidar = torch.from_numpy(origin_lidar)

                projected_lidar = cav_content['origin_lidar']
                projected_lidar[:, :3] = \
                    box_utils.project_points_by_matrix_torch(
                        projected_lidar[:, :3],
                        transformation_matrix)
                projected_lidar_list.append(projected_lidar)
                origin_cam_cur_agent = torch.from_numpy(
                    np.stack(origin_cam_cur_agent)).float().unsqueeze(0)
                output_dict[cav_id].update({'origin_lidar': origin_lidar,
                                            'origin_camera': origin_cam_cur_agent})
        if self.visualize:
            projected_lidar_list = projected_lidar_list[:1]
            if self.visualize_lidar_agent_only:
                projected_lidar_list = [projected_lidar_list[i] for i in
                                        range(len(projected_lidar_list)) if
                                        mode[i] == 1]
            projected_lidar_stack = [torch.from_numpy(
                np.vstack(projected_lidar_list))] if len(
                projected_lidar_list) else []
            output_dict['ego'].update(
                {'origin_lidar': projected_lidar_stack})

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
