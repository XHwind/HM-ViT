"""
Implementation of V2VNet Fusion
"""

import torch
import torch.nn as nn
from einops import rearrange

from opencood.models.sub_modules.torch_transformation_utils import \
    get_discretized_transformation_matrix, warp_affine, get_rotated_roi, \
    get_transformation_matrix
from opencood.models.sub_modules.convgru import ConvGRU
from opencood.models.sub_modules.spatial_transformation import \
    SpatialTransformation
from opencood.models.sub_modules.torch_transformation_utils import \
    get_roi_and_cav_mask

class V2VNetFusion(nn.Module):
    def __init__(self, args):
        super(V2VNetFusion, self).__init__()
        in_channels = args['in_channels']
        H, W = args['conv_gru']['H'], args['conv_gru']['W']
        kernel_size = args['conv_gru']['kernel_size']
        num_gru_layers = args['conv_gru']['num_layers']

        if 'resolution' in args:
            self.discrete_ratio = args['resolution']
        if 'voxel_size' in args:
            self.discrete_ratio = args['voxel_size'][0]

        self.downsample_rate = args['downsample_rate']

        self.spatial_transform = SpatialTransformation(args['spatial_transform'])

        self.num_iteration = args['num_iteration']
        self.gru_flag = args['gru_flag']
        self.agg_operator = args['agg_operator']

        self.msg_cnn = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3,
                                 stride=1, padding=1)
        self.conv_gru = ConvGRU(input_size=(H, W),
                                input_dim=in_channels * 2,
                                hidden_dim=[in_channels],
                                kernel_size=kernel_size,
                                num_layers=num_gru_layers,
                                batch_first=True,
                                bias=True,
                                return_all_layers=False)
        self.mlp = nn.Linear(in_channels, in_channels)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, mask, record_len, pairwise_t_matrix, prior_encoding=None):
        # x: (B,C,H,W)
        # record_len: (B)
        # pairwise_t_matrix: (B,L,L,4,4)
        # prior_encoding: (B,3)
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        # split x:[(L1, C, H, W), (L2, C, H, W)]
        split_x = self.regroup(x, record_len)

        batch_node_features = split_x
        # iteratively update the features for num_iteration times
        for l in range(self.num_iteration):

            batch_updated_node_features = []
            # iterate each batch
            for b in range(B):

                # number of valid agent
                N = record_len[b]
                # (N,N,4,4)
                updated_node_features = []
                # update each node i
                for i in range(N):

                    # flip the feature so the transformation is correct
                    batch_node_feature = batch_node_features[b]
                    transformation_matrix = pairwise_t_matrix[b,:N,i,...].unsqueeze(0)
                    # (N,C,H,W)
                    neighbor_feature = \
                        self.spatial_transform(batch_node_feature.unsqueeze(0),
                                               transformation_matrix).squeeze(0)
                    # (H,W,1,N)
                    com_mask = get_roi_and_cav_mask(
                        (1, N, H, W, batch_node_feature.shape[1]),
                        mask[b,:N].unsqueeze(0),
                        transformation_matrix,
                        self.discrete_ratio,
                        self.downsample_rate).squeeze(0)
                    # (N, 1, H, W)
                    com_mask = com_mask.permute(3, 2, 0, 1)
                    # (N,C,H,W)
                    ego_agent_feature = batch_node_feature[i].unsqueeze(
                        0).repeat(N, 1, 1, 1)
                    # (N,2C,H,W)
                    neighbor_feature = torch.cat(
                        [neighbor_feature, ego_agent_feature], dim=1)
                    # (N,C,H,W)
                    message = self.msg_cnn(neighbor_feature) * com_mask

                    # (C,H,W)
                    if self.agg_operator == "avg":
                        agg_feature = torch.mean(message, dim=0)
                    elif self.agg_operator == "max":
                        agg_feature = torch.max(message, dim=0)[0]
                    else:
                        raise ValueError("agg_operator has wrong value")
                    # (2C, H, W)
                    cat_feature = torch.cat(
                        [batch_node_feature[i, ...], agg_feature], dim=0)
                    # (C,H,W)
                    if self.gru_flag:
                        gru_out = \
                            self.conv_gru(
                                cat_feature.unsqueeze(0).unsqueeze(0))[
                                0][
                                0].squeeze(0).squeeze(0)
                    else:
                        gru_out = batch_node_feature[i, ...] + agg_feature
                    updated_node_features.append(gru_out.unsqueeze(0))
                # (N,C,H,W)
                batch_updated_node_features.append(
                    torch.cat(updated_node_features, dim=0))
            batch_node_features = batch_updated_node_features
        # (B,C,H,W)
        out = torch.cat(
            [itm[0, ...].unsqueeze(0) for itm in batch_node_features], dim=0)
        # (B,C,H,W)
        out = self.mlp(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return out
