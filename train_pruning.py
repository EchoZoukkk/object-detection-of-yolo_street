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
from models.pruning import prune_model
import os

def fine_tune_pruned(model_path, data_yaml='coco_street.yaml',
                     epochs=50, img_size=640, batch=8, name='finetune_640'):  # 改默认值
    """
    微调剪枝后的模型
    """
    model = YOLO(model_path)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        name=name,
        project='runs/prune',
        exist_ok=True,
        lr0=0.001,
        lrf=0.01,
        patience=15,
        warmup_epochs=5,
    )
    return model, results


def single_pruning(model_path, data_yaml='coco_street.yaml',
                   pruning_ratio=0.3, img_size=640, epochs=50):  # 改默认值
    print(f"\n{'=' * 60}")
    print(f"单次剪枝: 比例 {pruning_ratio}, 分辨率 {img_size}")
    print(f"{'=' * 60}")

    # 1. 剪枝
    print("=== 调用 prune_model 前 ===")
    try:
        model, orig_p, pruned_p = prune_model(model_path, pruning_ratio, img_size)
        print("=== prune_model 调用成功 ===")
        print(f"原始参数量: {orig_p / 1e6:.2f}M")
        print(f"剪枝后参数量: {pruned_p / 1e6:.2f}M")
    except Exception as e:
        print(f"=== prune_model 调用失败: {e} ===")
        import traceback
        traceback.print_exc()
        return

    # 2. 微调 - 文件夹名包含分辨率标识
    pruned_path = f'runs/prune/yolov8n_pruned_{int(pruning_ratio * 100)}.pt'
    print(f"pruned_path: {pruned_path}")

    print(f"\n{'=' * 60}")
    print(f"开始微调...")
    print(f"{'=' * 60}")

    # name 加上分辨率后缀，确保文件夹不同
    finetune_name = f'yolov8n_pruned_{int(pruning_ratio * 100)}_finetune_{img_size}'

    model, results = fine_tune_pruned(
        pruned_path,
        data_yaml,
        epochs=epochs,
        img_size=img_size,
        name=finetune_name
    )

    print(f"\n{'=' * 60}")
    print(f"剪枝完成！")
    print(f"  原始参数量: {orig_p / 1e6:.2f}M")
    print(f"  剪枝后参数量: {pruned_p / 1e6:.2f}M")
    print(f"  压缩率: {orig_p / pruned_p:.2f}x")
    print(f"  微调后模型: runs/prune/{finetune_name}/weights/best.pt")
    print(f"{'=' * 60}")

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='模型剪枝')
    parser.add_argument('--model', type=str,
                        default='best_640.pt',  # 改成 640
                        help='基线模型路径')
    parser.add_argument('--data', type=str, default='coco_street.yaml')
    parser.add_argument('--imgsz', type=int, default=640,  # 改成 640
                        help='输入分辨率')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--ratio', type=float, default=0.3,
                        help='剪枝比例 (0.2/0.3/0.5)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='微调轮数')

    args = parser.parse_args()

    single_pruning(
        args.model,
        args.data,
        pruning_ratio=args.ratio,
        img_size=args.imgsz,
        epochs=args.epochs
    )