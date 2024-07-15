# -*- coding: utf-8 -*-
"""
Unit test for pcd utilities
"""

import os
import sys
import unittest

import open3d as o3d

# temporary solution for relative imports in case opencood is not installed
# if opencda is installed, no need to use the following line
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from opencood.utils.pcd_utils import *


class TestPcdUtils(unittest.TestCase):
    def setUp(self):
        self.pcd_np = pcd_to_np('test/data/000147.pcd')
        assert len(self.pcd_np.shape) == 2
        assert self.pcd_np.shape[1] == 4

    def test_mask_out(self):
        filtered_pcd = \
            mask_points_by_range(self.pcd_np, [0, -50, -3, 100, 50, 1])

        assert filtered_pcd.shape[0] < self.pcd_np.shape[0]

        assert np.max(filtered_pcd, axis=0)[0] <= 100
        assert np.min(filtered_pcd, axis=0)[0] >= 0
        assert np.max(filtered_pcd, axis=0)[1] <= 50
        assert np.min(filtered_pcd, axis=0)[1] >= -50
        assert np.max(filtered_pcd, axis=0)[2] <= 50
        assert np.min(filtered_pcd, axis=0)[2] >= -50

    def test_ego_mask_out(self):
        filtered_pcd = \
            mask_ego_points(self.pcd_np)

        assert filtered_pcd.shape[0] <= self.pcd_np.shape[0]
