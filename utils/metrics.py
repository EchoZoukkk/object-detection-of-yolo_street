# Copyright 2026 Zou Kaiyu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
自定义评估指标计算
补充YOLOv8内置评估之外的指标
"""
import torch
import numpy as np
from collections import defaultdict


def compute_iou(box1, box2):
    """
    计算两个边界框的IoU
    box: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def compute_ap(recall, precision):
    """
    计算AP（Average Precision）
    使用11点插值法
    """
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0
    return ap


class COCOMetrics:
    """
    COCO风格评估指标
    """

    def __init__(self, num_classes=5, iou_thresholds=None):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7,
                                                 0.75, 0.8, 0.85, 0.9, 0.95]
        self.reset()

    def reset(self):
        self.predictions = defaultdict(list)  # {class_id: [(image_id, box, score)]}
        self.ground_truths = defaultdict(list)  # {class_id: [(image_id, box)]}
        self.gt_counter = defaultdict(lambda: defaultdict(int))  # 每张图的GT数量

    def add_batch(self, pred_boxes, pred_scores, pred_labels,
                  gt_boxes, gt_labels, image_ids):
        """
        添加一批预测和GT
        pred_boxes: [N, 4] 预测框 [x1,y1,x2,y2]
        pred_scores: [N] 置信度
        pred_labels: [N] 类别
        gt_boxes: [M, 4] GT框
        gt_labels: [M] GT类别
        image_ids: 图像ID列表
        """
        for i, img_id in enumerate(image_ids):
            # 筛选当前图像的预测
            pred_mask = [j for j, pid in enumerate(image_ids) if pid == img_id]
            gt_mask = [j for j, gid in enumerate(image_ids) if gid == img_id]

            for j in pred_mask:
                cls = int(pred_labels[j])
                self.predictions[cls].append((
                    img_id,
                    pred_boxes[j].cpu().numpy(),
                    float(pred_scores[j])
                ))

            for j in gt_mask:
                cls = int(gt_labels[j])
                self.ground_truths[cls].append((
                    img_id,
                    gt_boxes[j].cpu().numpy()
                ))
                self.gt_counter[cls][img_id] += 1

    def compute_map(self):
        """
        计算mAP
        """
        aps = []

        for cls in range(self.num_classes):
            cls_preds = self.predictions[cls]
            cls_gts = self.ground_truths[cls]

            if len(cls_gts) == 0:
                continue

            # 按置信度排序预测
            cls_preds.sort(key=lambda x: x[2], reverse=True)

            # 计算每个IoU阈值下的AP
            cls_aps = []
            for iou_thresh in self.iou_thresholds:
                tp = np.zeros(len(cls_preds))
                fp = np.zeros(len(cls_preds))

                # 记录每个GT是否被匹配
                gt_matched = defaultdict(list)

                for pred_idx, (img_id, pred_box, score) in enumerate(cls_preds):
                    # 找到同一张图的所有GT
                    img_gts = [(j, box) for j, (gid, box) in enumerate(cls_gts) if gid == img_id]

                    best_iou = 0
                    best_gt_idx = -1

                    for gt_j, gt_box in img_gts:
                        iou = compute_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_j

                    if best_iou >= iou_thresh and best_gt_idx not in gt_matched[img_id]:
                        tp[pred_idx] = 1
                        gt_matched[img_id].append(best_gt_idx)
                    else:
                        fp[pred_idx] = 1

                # 计算precision和recall
                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(fp)

                recalls = tp_cumsum / len(cls_gts)
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

                ap = compute_ap(recalls, precisions)
                cls_aps.append(ap)

            mean_ap = np.mean(cls_aps) if cls_aps else 0
            aps.append(mean_ap)

        return np.mean(aps) if aps else 0


def format_results(results_list):
    """
    格式化评估结果为表格字符串
    """
    header = f"{'Model':<25} {'Resolution':<12} {'mAP@0.5':<10} {'Params(M)':<12} {'FLOPs(G)':<12} {'Latency(ms)':<12} {'FPS':<8}"
    lines = [header, "-" * len(header)]

    for r in results_list:
        line = (f"{r['name']:<25} {r['img_size']:<12} {r['map50']:<10.4f} "
                f"{r['params_M']:<12.2f} {r['flops_G']:<12.2f} "
                f"{r['latency_ms']:<12.2f} {r['fps']:<8.2f}")
        lines.append(line)

    return "\n".join(lines)