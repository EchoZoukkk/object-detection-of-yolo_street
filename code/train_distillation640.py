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
YOLOv8 知识蒸馏脚本（拼写修正版）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


class DistillationTrainer(DetectionTrainer):
    """
    自定义蒸馏训练器
    """
    # 修正：rruns -> runs
    teacher_path = 'runs/detect/runs/train/yolov8m_street_teacher_416/weights/best.pt'
    distill_weight = 0.6
    temperature = 4.0

    def __init__(self, overrides=None, _callbacks=None):
        super().__init__(overrides=overrides, _callbacks=_callbacks)

        print(f"\n{'='*60}")
        print(f"[蒸馏] 教师模型: {self.teacher_path}")
        print(f"[蒸馏] 教师 mAP50: 0.607 | 基线 mAP50: ~0.587")
        print(f"[蒸馏] 温度 T={self.temperature}, 蒸馏权重 α={self.distill_weight}")
        print(f"{'='*60}")

        import os
        if not os.path.exists(self.teacher_path):
            raise FileNotFoundError(f"教师模型不存在: {self.teacher_path}")

        self.teacher = YOLO(self.teacher_path).model.to(self.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        print("[蒸馏] 教师模型加载完成，已冻结\n")

    def compute_loss(self, batch, preds=None):
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.model(batch['img'])

        loss, loss_items = self.criterion(preds, batch)

        if self.teacher is not None:
            with torch.no_grad():
                t_preds = self.teacher(batch['img'])

            distill_loss = 0.0
            T = self.temperature

            for s_pred, t_pred in zip(preds, t_preds):
                s_cls = s_pred[:, :, 4:]
                t_cls = t_pred[:, :, 4:]

                s_soft = F.log_softmax(s_cls / T, dim=-1)
                t_soft = F.softmax(t_cls / T, dim=-1)

                distill_loss += F.kl_div(
                    s_soft, t_soft, reduction='batchmean'
                ) * (T * T)

            loss = loss + self.distill_weight * distill_loss

        return loss, loss_items


def train_distillation():
    # 修正：rruns -> runs
    DistillationTrainer.teacher_path = 'runs/detect/runs/train/yolov8m_street_teacher_416/weights/best.pt'
    DistillationTrainer.distill_weight = 0.6
    DistillationTrainer.temperature = 4.0

    print("=" * 60)
    print("蒸馏训练：YOLOv8n <- YOLOv8m")
    print("=" * 60)

    student = YOLO('C:/ai_project/runs/detect/runs/train/yolov8n_street_distill_416/weights/last.pt')

    results = student.train(
        trainer=DistillationTrainer,
        data='coco_street.yaml',
        resume=True,
        epochs=100,
        imgsz=416,
        batch=8,
        device=0,
        amp=True,
        cos_lr=True,
        lr0=0.005,
        lrf=0.01,
        optimizer='SGD',
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3,
        name='yolov8n_street_distill_416',
        project='runs/train',
        exist_ok=True,
        patience=30,
        workers=4,
        cache=False,
        mosaic=1.0,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.6,
        scale=0.7,
        translate=0.2,
        copy_paste=0.1,
        mixup=0.1,
        plots=True,
        save=True,
    )

    print(f"\n蒸馏完成！")
    return student, results


if __name__ == '__main__':
    train_distillation()
