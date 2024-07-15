import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

from einops import rearrange

from opencood.models.sub_modules.torch_transformation_utils import \
    get_discretized_transformation_matrix, get_transformation_matrix, \
    warp_affine, get_rotated_roi
from opencood.utils.transformation_utils import x1_to_x2


def main():
    ego_image = cv2.imread(
        '/home/runshengxu/project/OpenCOOD/opv2v_data_dumping/tmp/2021_08_22_21_41_24/886/000069_bev_dynamic.png')
    cav_image = cv2.imread(
        '/home/runshengxu/project/OpenCOOD/opv2v_data_dumping/tmp/2021_08_22_21_41_24/904/000069_bev_dynamic.png')

    # resize image
    ego_image = np.array(cv2.resize(ego_image, (256, 256)),
                         dtype=np.float) / 255.
    ego_image[ego_image > 0] = 1
    cav_image = np.array(cv2.resize(cav_image, (256, 256)),
                         dtype=np.float) / 255.
    cav_image[cav_image > 0] = 1

    ego_image = torch.from_numpy(ego_image).cuda()
    cav_image = torch.from_numpy(cav_image).cuda()

    x = torch.zeros((2, 3, 256, 256)).cuda()
    x[0] = rearrange(ego_image, 'h w c -> c h w')
    x[1] = rearrange(cav_image, 'h w c -> c h w')

    plt.imshow(ego_image.cpu().numpy())
    plt.savefig(f"ego_origin.png")
    plt.clf()

    plt.imshow(cav_image.cpu().numpy())
    plt.savefig(f"cav_origin.png")
    plt.clf()

    ego_pose = [-123.2198715209961, 128.1273651123047, 0.03233739733695984,
                -0.11932373046875, -79.65458679199219, 0.19445547461509705]
    cav_pose = [-127.0864028930664, 91.43475341796875, 0.03186967596411705,
                0.041503533720970154, 92.17424011230469, 0.21723414957523346]
    transformation_matrix = x1_to_x2(cav_pose, ego_pose)

    # pairwise matrix cal
    pairwise_t_matrix = np.zeros((2, 2, 4, 4))
    # default are identity matrix
    pairwise_t_matrix[:, :] = np.identity(4)

    # return pairwise_t_matrix

    t_list = [x1_to_x2(ego_pose, ego_pose), transformation_matrix]

    for i in range(len(t_list)):
        for j in range(len(t_list)):
            # identity matrix to self
            if i == j:
                continue
            # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
            t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
            pairwise_t_matrix[i, j] = t_matrix
    pairwise_t_matrix = torch.from_numpy(pairwise_t_matrix).cuda()
    B = 1
    L = 2
    H = 256
    W = 256
    pairwise_t_matrix = get_discretized_transformation_matrix(
        pairwise_t_matrix.reshape(-1, L, 4, 4), 0.390625,
        1).reshape(B, L, L, 2, 3)
    # (B*L,L,1,H,W)
    roi_mask = get_rotated_roi((B * L, L, 1, H, W),
                               pairwise_t_matrix.reshape(B * L * L, 2, 3))
    roi_mask = roi_mask.reshape(B, L, L, 1, H, W)

    b = 0
    N = 2
    t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
    for i in range(N):
        # (N,1,H,W)
        mask = roi_mask[b, :N, i, ...]

        # flip the feature so the transformation is correct
        batch_node_feature = x
        batch_node_feature = rearrange(batch_node_feature,
                                       'b c h w  -> b c w h')
        batch_node_feature = torch.flip(batch_node_feature,
                                        dims=(3,))
        current_t_matrix = t_matrix[:, i, :, :]
        current_t_matrix = get_transformation_matrix(current_t_matrix, (H, W))
        # (N,C,H,W)
        neighbor_feature = warp_affine(batch_node_feature,
                                       current_t_matrix,
                                       (H, W))

        neighbor_feature = torch.flip(neighbor_feature,
                                          dims=(3,))
        neighbor_feature = rearrange(neighbor_feature,
                                         'b c w h -> b c h w ')

        plt.imshow(neighbor_feature[0].permute(1, 2, 0).cpu().numpy())
        plt.savefig(f"ego_new_{i}.png")
        plt.clf()

        plt.imshow(neighbor_feature[1].permute(1, 2, 0).cpu().numpy())
        plt.savefig(f"cav_new_{i}.png")
        plt.clf()

    print('test')


if __name__ == '__main__':
    main()
