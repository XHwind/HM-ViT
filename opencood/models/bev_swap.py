import torch
from torch import nn
from einops import rearrange

from opencood.models.fusion_modules.swap_fusion_modules import \
    SwapFusionEncoder
from opencood.models.backbones.resnet_encoder_concat import\
    ResnetEncoderConcat
from opencood.models.sub_modules.naive_decoder import NaiveDecoder


class BevSwap(nn.Module):
    def __init__(self, params):
        super(BevSwap, self).__init__()

        # encoder params
        self.encoder = ResnetEncoderConcat(params['encoder'])

        # view fuse network
        self.vfn = SwapFusionEncoder(params['view_swap_fuse'])

        # decoder
        self.decoder = NaiveDecoder(params['decoder'])

        # segmentation head
        self.static_head = nn.Conv2d(params['seg_head_dim'],
                                     params['output_class'],
                                     kernel_size=3,
                                     padding=1)
        self.dynamic_head = nn.Conv2d(params['seg_head_dim'],
                                      params['output_class'],
                                      kernel_size=3,
                                      padding=1)

    def forward(self, batch_dict):

        # shape: (B,L,M,H,W,3)
        x = batch_dict['inputs']
        b, l = x.shape[0], x.shape[1]

        # (B, L, M, C, H, W)
        x = self.encoder(x)
        x = rearrange(x, 'b l m c h w -> (b l) m c h w')

        # (B, C, H, W)
        x = self.vfn(x)
        # (B, L, C, H, W)
        x = rearrange(x, '(b l) c h w -> b l c h w', b=b, l=l)
        x = self.decoder(x)

        # reshape to correct format
        b, l, c, h, w = x.shape
        x = rearrange(x, 'b l c h w -> (b l) c h w')

        static_map = self.static_head(x)
        dynamic_map = self.dynamic_head(x)

        static_map = rearrange(static_map, '(b l) c h w -> b l c h w',
                               l=l)
        dynamic_map = rearrange(dynamic_map, '(b l) c h w -> b l c h w',
                                l=l)

        output_dict = {'static_seg': static_map,
                       'dynamic_seg': dynamic_map}

        return output_dict


if __name__ == '__main__':
    import os
    from opencood.hypes_yaml.yaml_utils import load_yaml

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    test_data = torch.rand(1, 1, 4, 512, 512, 3)
    test_data = test_data.cuda()

    params = load_yaml('../hypes_yaml/opcamera/bev_swap.yaml')

    model = BevSwap(params['model']['args'])
    model = model.cuda()
    while True:
        output = model({'inputs': test_data})
        print('test_passed')
