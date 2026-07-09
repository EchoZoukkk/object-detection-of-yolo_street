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
COCO 转 YOLO 格式，只保留街景类别
"""

import json
import os
from pathlib import Path
from tqdm import tqdm

# COCO 街景类别映射（原始ID -> 新ID）
STREET_CATEGORIES = {
    1: 0,  # person
    3: 1,  # car
    6: 2,  # bus
    8: 3,  # truck
    10: 4,  # traffic light
    13: 5,  # stop sign
    4: 6,  # motorcycle
    2: 7,  # bicycle
}


def convert_coco_to_yolo(data_root, split='train2017'):
    """
    转换 COCO 标注为 YOLO 格式
    """
    data_root = Path(data_root)

    # 路径
    img_dir = data_root / split
    ann_file = data_root / 'annotations' / f'instances_{split}.json'
    label_dir = data_root / 'labels' / split.replace('2017', '')

    # 创建标签目录
    label_dir.mkdir(parents=True, exist_ok=True)

    # 加载 COCO 标注
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)

    # 建立图像ID到文件名和尺寸的映射
    images = {img['id']: img for img in coco_data['images']}

    # 按图像分组标注
    img_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    # 转换每张图像的标注
    converted_count = 0
    skipped_count = 0

    for img_id, img_info in tqdm(images.items(), desc=f"转换 {split}"):
        img_w = img_info['width']
        img_h = img_info['height']
        img_name = Path(img_info['file_name']).stem  # 去掉扩展名

        # 获取该图的所有标注
        anns = img_annotations.get(img_id, [])

        # 筛选街景类别并转换
        yolo_labels = []
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id not in STREET_CATEGORIES:
                continue

            # COCO bbox: [x, y, w, h]（左上角坐标+宽高）
            x, y, w, h = ann['bbox']

            # 转换为 YOLO 格式：[class, x_center, y_center, width, height]（归一化）
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            width = w / img_w
            height = h / img_h

            # 裁剪到 [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))

            new_class_id = STREET_CATEGORIES[cat_id]
            yolo_labels.append(f"{new_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # 写入标签文件
        label_file = label_dir / f"{img_name}.txt"
        if yolo_labels:
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_labels))
            converted_count += 1
        else:
            # 没有街景目标的图像，创建空文件或跳过
            with open(label_file, 'w') as f:
                pass  # 空文件
            skipped_count += 1

    print(f"\n{split} 转换完成:")
    print(f"  有街景目标的图像: {converted_count}")
    print(f"  无街景目标的图像: {skipped_count}")
    print(f"  标签保存至: {label_dir}")


if __name__ == '__main__':
    DATA_ROOT = r"C:\Users\23564\Desktop\object_detection\data"

    # 转换训练集和验证集
    convert_coco_to_yolo(DATA_ROOT, 'train2017')
    convert_coco_to_yolo(DATA_ROOT, 'val2017')

    print("\n全部转换完成！")
    print("请更新 coco_street.yaml 使用新的路径格式")