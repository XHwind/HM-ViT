import argparse
import time
import os

import torch
import open3d as o3d
import tqdm
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, infrence_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import vis_utils
from opencood.utils import eval_utils


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=False,
                        help='Continued training path')
    parser.add_argument('--epoch', required=False, type=int,
                        default=-1,
                        help='which epoch to load for inference')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='nofusion, late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--mixed_fusion', action='store_true',
                        help='whether to load both camera and lidar models '
                             'for no/late fusion')
    parser.add_argument('--camera_model_dir', type=str, required=False,
                        help='Camera model path for mixed late fusion')
    parser.add_argument('--lidar_model_dir', type=str, required=False,
                        help='LiDAR model path for mixed late fusion')
    parser.add_argument('--ap_mode', type=str, required=False,
                        help='Average precision mode (iou, distance, both)')
    parser.add_argument('--camera_to_lidar_ratio', type=str, required=False,
                        help='Camera to LiDAR ratio')
    parser.add_argument('--ego_mode', type=str, required=False,
                        help="Ego mode. Please choose from [camera, lidar, mixed]")
    parser.add_argument('--save_vis_name', type=str, required=False,
                        default='vis')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()

    if opt.mixed_fusion:
        assert opt.camera_model_dir and opt.lidar_model_dir, \
            "For mixed camera and LiDAR mode, camera_model_dir and lidar_model_dir" \
            "must be set."
        file_camera = os.path.join(opt.camera_model_dir, 'config.yaml')
        file_lidar = os.path.join(opt.lidar_model_dir, 'config.yaml')
        # Use camera hypes to set dataset, camera/lidar ratio etc.
        hypes = hypes_camera = yaml_utils.load_yaml(file_camera)
        hypes_lidar = yaml_utils.load_yaml(file_lidar)
    else:
        hypes = yaml_utils.load_yaml(None, opt)
    if opt.camera_to_lidar_ratio:
        hypes['camera_to_lidar_ratio'] = float(opt.camera_to_lidar_ratio)
    if opt.ego_mode:
        hypes['ego_mode'] = opt.ego_mode

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=10,
                             collate_fn=opencood_dataset.collate_batch,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Creating Model')
    if opt.mixed_fusion:
        model_camera = train_utils.create_model(hypes_camera)
        model_lidar = train_utils.create_model(hypes_lidar)
        model_camera.to(device)
        model_lidar.to(device)
    else:
        model = train_utils.create_model(hypes)
        # we assume gpu is necessary
        model.to(device)

    print('Loading Model from checkpoint')
    if opt.mixed_fusion:
        _, model_camera = train_utils.load_saved_model(opt.camera_model_dir,
                                                       model_camera)
        _, model_lidar = train_utils.load_saved_model(opt.lidar_model_dir,
                                                      model_lidar)
        model_camera.eval()
        model_lidar.eval()
    else:
        _, model = train_utils.load_saved_model(opt.model_dir, model,
                                                epoch=opt.epoch)
        model.eval()

    # Create the dictionary for evaluation
    result_stat = {}
    threshs_dict = {
        'iou': [0.3, 0.5, 0.7],
        'distance': [0.5, 1.0, 2.0, 4.0]
    }
    if opt.ap_mode == 'iou':
        result_stat['iou'] = {}
    elif opt.ap_mode == 'distance':
        result_stat['distance'] = {}
    else:
        result_stat['iou'] = {}
        result_stat['distance'] = {}

    for mode in result_stat:
        threshs = threshs_dict[mode]
        for th in threshs:
            result_stat[mode].update({th: {'tp': [], 'fp': [], 'gt': 0}})

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(50):
            vis_aabbs_gt.append(o3d.geometry.LineSet())
            vis_aabbs_pred.append(o3d.geometry.LineSet())
    pbar = tqdm.tqdm(total=len(data_loader), leave=True)
    for i, batch_data in enumerate(data_loader):
        with torch.no_grad():
            torch.cuda.synchronize()

            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'nofusion' and not opt.mixed_fusion:
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_no_fusion(batch_data,
                                                       model,
                                                       opencood_dataset)
            elif opt.fusion_method == 'nofusion' and opt.mixed_fusion:
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_mixed_no_fusion(batch_data,
                                                             model_camera,
                                                             model_lidar,
                                                             opencood_dataset)
            elif opt.fusion_method == 'late' and not opt.mixed_fusion:
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_late_fusion(batch_data,
                                                         model,
                                                         opencood_dataset)
            elif opt.fusion_method == 'late' and opt.mixed_fusion:
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_mixed_late_fusion(batch_data,
                                                               model_camera,
                                                               model_lidar,
                                                               opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_early_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_intermediate_fusion(batch_data,
                                                                 model,
                                                                 opencood_dataset)
            else:
                raise NotImplementedError(
                    'Only nofusion, late early, and intermediate'
                    'fusion is supported.')

            for mode in result_stat.keys():
                threshs = threshs_dict[mode]
                for thresh in threshs:
                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                               pred_score,
                                               gt_box_tensor,
                                               result_stat[mode],
                                               thresh,
                                               mode=mode)

            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                infrence_utils.save_prediction_gt(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'][0],
                                                  i,
                                                  npy_save_path)
            pbar.update(1)
            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir,
                                                 opt.save_vis_name)
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)
                origin_lidar = batch_data['ego']['origin_lidar'][0] if len(
                    batch_data['ego']['origin_lidar']) else torch.Tensor([]).to(pred_box_tensor.device)
                opencood_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  origin_lidar,
                                                  opt.show_vis,
                                                  vis_save_path,
                                                  dataset=opencood_dataset)

            if opt.show_sequence:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'][0],
                        vis_pcd,
                        mode='constant'
                    )
                if i == 0:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_pred,
                                                 pred_o3d_box,
                                                 update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_gt,
                                                 gt_o3d_box,
                                                 update_mode='add')

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box)
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir,
                                  mode=opt.ap_mode)
    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()
