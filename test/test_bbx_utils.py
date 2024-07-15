# -*- coding: utf-8 -*-
"""
Unit test for bbx utilities.
"""

import os
import sys
import unittest

import torch
import cv2
import matplotlib.pyplot as plt

import open3d as o3d

# temporary solution for relative imports in case opencood is not installed
# if opencda is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from opencood.utils.box_utils import *


class TestBbxUtils(unittest.TestCase):
    def setUp(self):
        self.bbx_center = np.array(
            [[132.17, 118.69, 0.75, 2.256 * 2, 1.003 * 2, 0.762 * 2, 1.2]])
        self.objects = np.array([[30, 60, 2, 3, 5, 3, 0],
                                 [70, -20, 2, 3, 5, 3, 0.5],
                                 [120, 0, 2, 3, 5, 3, 0.7],
                                 [30, -80, 2, 3, 5, 3, -0.2],
                                 [130, 60, 2, 3, 5, 3, 0],
                                 [50, 61, 2, 3, 5, 3, 0]])
        from opencood.hypes_yaml.yaml_utils import load_yaml
        self.params = load_yaml('test/data/000147.yaml')
        self.order = 'lwh'

    def test_center2corner2center(self):
        corners = boxes_to_corners_3d(self.bbx_center, order=self.order)
        bbx_center = corner_to_center(corners, order=self.order)
        assert np.abs(np.sum(self.bbx_center - bbx_center)) < 0.01

    def test_center2corner2center_stress(self):
        for i in range(100):
            noise = np.random.random((1, 7)) * 3
            noise_bbx_center = self.bbx_center + noise

            while noise_bbx_center[0, 6] > np.pi:
                noise_bbx_center[0, 6] -= np.pi

            corners = boxes_to_corners_3d(noise_bbx_center, self.order)
            bbx_center = corner_to_center(corners, self.order)
            assert np.abs(np.sum(noise_bbx_center - bbx_center)) < 0.01

    def test_center2corner2center_n(self):
        random_bbx = np.random.random((100, 7))
        corner = boxes_to_corners_3d(random_bbx, self.order)
        bbx_center = corner_to_center(corner, order=self.order)
        assert np.abs(np.sum(random_bbx - bbx_center)) < 0.01

    def test_bbx_mask_out(self):
        mask_out_bbx = mask_boxes_outside_range_numpy(self.objects,
                                                      [0, -60, -1, 60, 70, 3],
                                                      self.order)
        assert mask_out_bbx.shape[0] == 2
        assert mask_out_bbx.shape[1] == 7

    def test_project_objects(self):
        objects = self.params['vehicles']
        lidar_pose = self.params['lidar_pose']
        output_dict = {}
        lidar_range = [0, -50, -3, 100, 50, 1]

        project_world_objects(objects, output_dict, lidar_pose, lidar_range,
                        self.order)

        assert len(output_dict) > 0
        for key, val in output_dict.items():
            assert val.shape == (1, 7)

    def test_nms(self):
        objects = np.array([[0, 10, 10, 20, 0.5],
                            [1, 7, 5, 15, 0.9],
                            [10, 20, 30, 40, 0.5]])
        objects = torch.from_numpy(objects)
        keep_index = nms_pytorch(objects, 0.1)
        print(keep_index)
        assert len(keep_index) == 2
