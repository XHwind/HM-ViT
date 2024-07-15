import os

import numpy as np
import torch

from opencood.utils import common_utils
from opencood.hypes_yaml import yaml_utils
from opencood.utils.box_utils import corner_to_center


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, thresh,
                    mode='iou'):
    """
    Calculate the true positive and false positive numbers of the current
    frames.

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    thresh : float
        The iou thresh.
    mode : str
        mode for calculating tp and fp, chosen from 'iou' and 'distance'
    """
    if mode == 'iou':
        caluclate_tp_fp_iou(det_boxes, det_score, gt_boxes, result_stat,
                            thresh)
    elif mode == 'distance':
        caluclate_tp_fp_distance(det_boxes, det_score, gt_boxes, result_stat,
                                 thresh)
    else:
        raise ValueError(f"Mode must be either iou or distance but received "
                         f"{mode}.")


def center_distance(gt_box, pred_box):
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box[:2]) - np.array(gt_box[:2]))


def caluclate_tp_fp_distance(pred_boxes, pred_score, gt_boxes, result_stat,
                             dist_th):
    """
    Calculate the true positive and false positive numbers of the current
    frames.

    Parameters
    ----------
    pred_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    pred_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    dist_th : float
        The distance thresh in meters.
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    taken = set()
    if pred_boxes is not None:
        # convert bounding boxes to numpy array
        pred_boxes = common_utils.torch_tensor_to_numpy(pred_boxes)
        pred_score = common_utils.torch_tensor_to_numpy(pred_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        pred_boxes = corner_to_center(pred_boxes)
        gt_boxes = corner_to_center(gt_boxes)

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-pred_score)
        dist_matrix = np.linalg.norm(gt_boxes[np.newaxis, :, :2] - pred_boxes[:, np.newaxis, :2], axis=-1)
        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            pred_idx = score_order_descend[i]
            pred_box = pred_boxes[pred_idx]
            min_dist = np.inf
            match_gt_idx = None
            for gt_idx in range(len(gt_boxes)):
                gt_box = gt_boxes[gt_idx]
                if gt_idx not in taken:
                    # this_distance = center_distance(gt_box, pred_box)
                    this_distance = dist_matrix[pred_idx, gt_idx]
                    if this_distance < min_dist:
                        min_dist = this_distance
                        match_gt_idx = gt_idx
            # If the closest match is close enough according to threshold we have a match!
            is_match = min_dist < dist_th
            if is_match:
                taken.add(match_gt_idx)
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

    result_stat[dist_th]['fp'] += fp
    result_stat[dist_th]['tp'] += tp
    result_stat[dist_th]['gt'] += gt


def caluclate_tp_fp_iou(det_boxes, det_score, gt_boxes, result_stat,
                        iou_thresh):
    """
    Calculate the true positive and false positive numbers of the current
    frames.

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))

        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)

    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt


def calculate_ap(result_stat, iou):
    """
    Calculate the average precision and recall, and save them into a txt.

    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
    iou : float
    """
    iou_5 = result_stat[iou]

    fp = iou_5['fp']
    tp = iou_5['tp']
    assert len(fp) == len(tp)

    gt_total = iou_5['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results(result_stat, save_path, mode='iou'):
    dump_dict = {}
    print("")
    for mode in result_stat.keys():
        dump_dict[mode] = {}
        if mode == 'iou':
            ap_30, mrec_30, mpre_30 = calculate_ap(result_stat[mode], 0.30)
            ap_50, mrec_50, mpre_50 = calculate_ap(result_stat[mode], 0.50)
            ap_70, mrec_70, mpre_70 = calculate_ap(result_stat[mode], 0.70)

            dump_dict[mode].update({'ap_30': ap_30,
                              'ap_50': ap_50,
                              'ap_70': ap_70,
                              'mpre_50': mpre_50,
                              'mrec_50': mrec_50,
                              'mpre_70': mpre_70,
                              'mrec_70': mrec_70,
                              })
            print('AP@0.3 is %.3f\n'
                  'AP@0.5 is %.3f\n'
                  'AP@0.7 is %.3f' % (
                      ap_30, ap_50, ap_70))
        if mode == 'distance':
            msg = ""
            aps = []
            for th in result_stat[mode].keys():
                ap, mrec, mpre = calculate_ap(result_stat[mode], th)
                aps.append(ap)
                ap_key = "ap_" + str(th)
                mrec_key = "mrec_" + str(th)
                mpre_key = "mpre_" + str(th)
                dump_dict[mode].update({ap_key: ap,
                                  mpre_key: mpre,
                                  mrec_key: mrec
                                  })
                msg += f"dAP@ {th} is {ap:.3f}\n"
            mAP = np.mean(aps)
            dump_dict[mode].update({"map": mAP.item()})
            msg += f"mAP is {mAP:.3f}"
            print(msg)

    yaml_utils.save_yaml(dump_dict, os.path.join(save_path, 'eval.yaml'))
