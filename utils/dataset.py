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
COCO数据集处理：室外街景检测专用
策略：筛选目标类别 + 场景过滤 + 数据增强
"""

import os
import json
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import cv2
import numpy as np
from pathlib import Path
import random


# 室外街景相关类别（COCO原始ID -> 新ID）
STREET_CATEGORIES = {
    1: 0,   # person -> 0
    3: 1,   # car -> 1
    6: 2,   # bus -> 2
    8: 3,   # truck -> 3
    10: 4,  # traffic light -> 4
    13: 5,  # stop sign -> 5
    4: 6,   # motorcycle -> 6
    2: 7,   # bicycle -> 7
}

# 室内常见类别（用于排除室内场景）
INDOOR_CATEGORIES = {
    62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
    67: 'dining table', 72: 'tv', 73: 'laptop', 74: 'mouse',
    75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
    79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
}


class COCOStreetDataset(Dataset):
    """
    COCO街景数据集：筛选室外场景图像
    """

    def __init__(self, root_dir, ann_file, img_size=640, mode='train2017',
                 scene_filter=True, augment_street=True):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.mode = mode
        self.scene_filter = scene_filter
        self.augment_street = augment_street

        self.coco = COCO(ann_file)

        # 筛选有效图像
        self.valid_images = self._filter_images()
        self.image_ids = list(self.valid_images.keys())

        print(f"[Dataset] {mode}: {len(self.image_ids)} 张有效图像")

        # 打印类别分布
        self._print_class_distribution()

    def _filter_images(self):
        """
        筛选图像：包含街景目标 + 排除明显室内场景
        """
        valid = {}

        for img_id in self.coco.imgs.keys():
            # 获取该图的所有标注
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)

            # 检查是否包含室内物品（如果包含且比例高，可能是室内场景）
            indoor_count = 0
            street_count = 0
            street_anns = []

            for ann in anns:
                cat_id = ann['category_id']

                if cat_id in INDOOR_CATEGORIES:
                    indoor_count += 1
                elif cat_id in STREET_CATEGORIES:
                    street_count += 1
                    street_anns.append(ann)

            # 筛选条件：
            # 1. 至少包含1个街景目标
            # 2. 街景目标数量 > 室内目标数量（简单场景过滤）
            # 3. 或者没有室内目标
            if street_count > 0:
                if not self.scene_filter or indoor_count <= street_count:
                    valid[img_id] = street_anns

        return valid

    def _print_class_distribution(self):
        """打印类别分布统计"""
        class_counts = {cat_name: 0 for cat_name in STREET_CATEGORIES.values()}

        for img_id, anns in self.valid_images.items():
            for ann in anns:
                new_id = STREET_CATEGORIES[ann['category_id']]
                class_counts[new_id] += 1

        # 类别名称映射
        id_to_name = {v: k for k, v in {
            'person': 0, 'car': 1, 'bus': 2, 'truck': 3,
            'traffic_light': 4, 'stop_sign': 5, 'motorcycle': 6, 'bicycle': 7
        }.items()}

        print("\n类别分布:")
        for cls_id, count in sorted(class_counts.items()):
            print(f"  {id_to_name.get(cls_id, cls_id)}: {count}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        # 读取图像
        img_path = self.root_dir / img_info['file_name']
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img.shape[:2]

        # 读取标注（已筛选的街景目标）
        anns = self.valid_images[img_id]

        labels = []
        for ann in anns:
            new_cat_id = STREET_CATEGORIES[ann['category_id']]
            x, y, w, h = ann['bbox']

            # 归一化
            x_center = (x + w / 2) / orig_w
            y_center = (y + h / 2) / orig_h
            width = w / orig_w
            height = h / orig_h

            labels.append([new_cat_id, x_center, y_center, width, height])

        # 街景数据增强（模拟不同光照/天气条件）
        if self.augment_street and self.mode == 'train2017':
            img = self._street_augmentation(img)

        # 图像预处理
        img, ratio, pad = self._letterbox(img, self.img_size)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float() / 255.0

        # 调整标签
        labels = np.array(labels) if labels else np.zeros((0, 5))
        if len(labels) > 0:
            labels[:, 1] = labels[:, 1] * ratio[0] + pad[0] / self.img_size
            labels[:, 2] = labels[:, 2] * ratio[1] + pad[1] / self.img_size
            labels[:, 3] *= ratio[0]
            labels[:, 4] *= ratio[1]

        labels = torch.from_numpy(labels).float()

        return img, labels, img_id

    def _street_augmentation(self, img):
        """
        街景专用数据增强：模拟不同天气/光照条件
        """
        aug_type = random.choice(['none', 'rain', 'night', 'fog', 'sun'])

        if aug_type == 'none':
            return img

        elif aug_type == 'rain':
            # 模拟雨天：降低亮度 + 添加噪声
            img = cv2.convertScaleAbs(img, alpha=0.8, beta=0)
            noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)

        elif aug_type == 'night':
            # 模拟夜晚：大幅降低亮度 + 提高对比度
            img = cv2.convertScaleAbs(img, alpha=0.4, beta=-30)

        elif aug_type == 'fog':
            # 模拟雾天：降低对比度 + 雾化效果
            img = cv2.convertScaleAbs(img, alpha=0.6, beta=20)
            blur = cv2.GaussianBlur(img, (15, 15), 0)
            img = cv2.addWeighted(img, 0.7, blur, 0.3, 0)

        elif aug_type == 'sun':
            # 模拟强光：提高亮度 + 轻微过曝
            img = cv2.convertScaleAbs(img, alpha=1.2, beta=30)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        """保持纵横比的resize"""
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=color)

        return img, (r, r), (dw, dh)