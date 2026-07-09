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
from ultralytics import YOLO

def train_baseline(data_yaml, epochs=100, img_size=640, batch=16, suffix='640'):
    model = YOLO('C:/ai_project/runs/detect/runs/train/yolov8n_street_416/weights/last.pt')  # 官方预训练权重

    results = model.train(
        data=data_yaml,
        resume=True,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        device=0,
        amp=True,

        # 数据增强（针对街景优化）
        mosaic=1.0,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.6,          # 增强亮度变化
        scale=0.7,          # 增强尺度变化
        translate=0.2,      # 增强位置偏移
        copy_paste=0.1,     # 模拟遮挡
        mixup=0.1,          # 边界模糊鲁棒性

        # 学习率优化
        cos_lr=True,        # 余弦退火
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3.0,

        # 效率优化
        workers=8,          # 根据CPU调整
        cache=False,       # 或 True，加速数据加载

        # 保存配置
        name=f'yolov8n_street_{suffix}',
        project='runs/train',
        exist_ok=True,
        patience=30,        # 对应100 epoch
        save=True,
        plots=True,
    )
    return model, results

if __name__ == '__main__':
    model_416, _ = train_baseline('coco_street.yaml', 100, 416, 16, '416')
