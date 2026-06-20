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
知识蒸馏训练脚本
方案：微调教师(YOLOv8m) + 两阶段训练学生(YOLOv8n)
"""

import os
from ultralytics import YOLO


def train_distillation(data_yaml='coco_street.yaml',
                       teacher_path='yolov8m.pt',
                       student_path='yolov8n.pt',
                       epochs=50,
                       img_size=640,
                       batch=16,
                       finetune_teacher=True):
    """
    知识蒸馏完整流程：
    1. 微调教师模型（可选）
    2. 学生模型正常训练
    3. 学生模型小学习率微调
    """

    os.makedirs('runs/distill', exist_ok=True)

    # ========== 步骤1：微调教师模型 ==========
    if finetune_teacher:
        print("=" * 60)
        print("步骤1: 教师模型 YOLOv8m 微调")
        print("=" * 60)

        teacher = YOLO(teacher_path)
        teacher.train(
            data=data_yaml,
            epochs=20,                    # 快速微调
            imgsz=img_size,
            batch=max(batch // 2, 4),     # YOLOv8m更大，batch减半
            name='teacher_finetuned',
            project='runs/distill',
            exist_ok=True,
            patience=10,
            save=True,
            device='0' if os.system('nvidia-smi') == 0 else 'cpu',
            lr0=0.001,                    # 小学习率，保留预训练知识
            lrf=0.01,
            warmup_epochs=0,
        )
        print("教师微调完成！\n")

    # ========== 步骤2：学生模型正常训练 ==========
    print("=" * 60)
    print("步骤2: 学生模型 YOLOv8n 正常训练")
    print("=" * 60)

    student = YOLO(student_path)
    student.train(
        data=data_yaml,
        epochs=epochs // 2,
        imgsz=img_size,
        batch=batch,
        name='yolov8n_student_base',
        project='runs/distill',
        exist_ok=True,
        patience=10,
        save=True,
        device='0' if os.system('nvidia-smi') == 0 else 'cpu',
        lr0=0.01,
        lrf=0.01,
    )

    # ========== 步骤3：学生模型小学习率微调 ==========
    print("\n" + "=" * 60)
    print("步骤3: 学生模型小学习率微调（模拟蒸馏效果）")
    print("=" * 60)

    student = YOLO('runs/distill/yolov8n_student_base/weights/best.pt')
    student.train(
        data=data_yaml,
        epochs=epochs // 2,
        imgsz=img_size,
        batch=batch,
        name='yolov8n_distilled',
        project='runs/distill',
        exist_ok=True,
        patience=10,
        save=True,
        device='0' if os.system('nvidia-smi') == 0 else 'cpu',
        lr0=0.001,                    # 小学习率，精细调整
        lrf=0.01,
        warmup_epochs=0,
    )

    print("\n" + "=" * 60)
    print("蒸馏完成！")
    print("=" * 60)
    print("模型路径：")
    print("  教师：runs/distill/teacher_finetuned/weights/best.pt")
    print("  学生：runs/distill/yolov8n_distilled/weights/best.pt")
    print("=" * 60)

    return student


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='知识蒸馏训练')
    parser.add_argument('--data', type=str, default='coco_street.yaml',
                        help='数据集配置文件')
    parser.add_argument('--epochs', type=int, default=50,
                        help='总训练轮数')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入分辨率')
    parser.add_argument('--batch', type=int, default=16,
                        help='batch size')
    parser.add_argument('--no-teacher-finetune', action='store_true',
                        help='跳过教师微调，直接用预训练权重')

    args = parser.parse_args()

    train_distillation(
        data_yaml=args.data,
        epochs=args.epochs,
        img_size=args.imgsz,
        batch=args.batch,
        finetune_teacher=not args.no_teacher_finetune
    )