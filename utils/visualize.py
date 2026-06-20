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
可视化工具：绘制对比图、特征图、检测结果等
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from pathlib import Path


def plot_comparison_curves(results, save_path='comparison_results.png'):
    """
    绘制精度-速度权衡曲线（evaluate.py中调用的核心可视化）
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 按分辨率分组
    res_416 = [r for r in results if r['img_size'] == 416]
    res_640 = [r for r in results if r['img_size'] == 640]

    colors = {'YOLOv8n Baseline': '#1f77b4',
              'YOLOv8n Distilled': '#ff7f0e',
              'YOLOv8n Pruned': '#2ca02c',
              'YOLOv8n Quantized': '#d62728'}

    # 1. mAP vs FLOPs
    ax = axes[0, 0]
    for res, marker, size_label in [(res_416, 'o', '416'), (res_640, 's', '640')]:
        for r in res:
            name = r['name'].split('_')[0]
            color = colors.get(name, '#333333')
            ax.scatter(r['flops_G'], r['map50'], c=color, marker=marker,
                       s=150, edgecolors='black', linewidth=1.5, zorder=5)
            ax.annotate(f"{name}\n({size_label})",
                        (r['flops_G'], r['map50']),
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=8)

    ax.set_xlabel('FLOPs (G)', fontsize=12)
    ax.set_ylabel('mAP@0.5', fontsize=12)
    ax.set_title('Accuracy vs Computational Cost', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # 添加图例说明分辨率
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=10, label='416x416'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markersize=10, label='640x640')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # 2. mAP vs 参数量
    ax = axes[0, 1]
    for res, marker in [(res_416, 'o'), (res_640, 's')]:
        for r in res:
            name = r['name'].split('_')[0]
            color = colors.get(name, '#333333')
            ax.scatter(r['params_M'], r['map50'], c=color, marker=marker,
                       s=150, edgecolors='black', linewidth=1.5)
            ax.annotate(name.split()[-1], (r['params_M'], r['map50']),
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=9)

    ax.set_xlabel('Parameters (M)', fontsize=12)
    ax.set_ylabel('mAP@0.5', fontsize=12)
    ax.set_title('Accuracy vs Model Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # 3. mAP vs Latency (核心：精度-速度权衡)
    ax = axes[1, 0]
    for res, marker in [(res_416, 'o'), (res_640, 's')]:
        for r in res:
            name = r['name'].split('_')[0]
            color = colors.get(name, '#333333')
            ax.scatter(r['latency_ms'], r['map50'], c=color, marker=marker,
                       s=150, edgecolors='black', linewidth=1.5)

    ax.set_xlabel('Latency (ms)', fontsize=12)
    ax.set_ylabel('mAP@0.5', fontsize=12)
    ax.set_title('Accuracy-Speed Trade-off Curve', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # 添加帕累托前沿线
    all_points = [(r['latency_ms'], r['map50']) for r in results]
    all_points.sort()
    pareto_x, pareto_y = [all_points[0][0]], [all_points[0][1]]
    for x, y in all_points[1:]:
        if y > pareto_y[-1]:
            pareto_x.append(x)
            pareto_y.append(y)
    ax.plot(pareto_x, pareto_y, 'r--', alpha=0.5, linewidth=2, label='Pareto Frontier')
    ax.legend()

    # 4. 分辨率影响柱状图
    ax = axes[1, 1]
    model_names = []
    map_416_list = []
    map_640_list = []

    for r in results:
        base_name = r['name'].split('_')[0]
        if base_name not in model_names:
            model_names.append(base_name)

    x = np.arange(len(model_names))
    width = 0.35

    for name in model_names:
        m416 = next((r['map50'] for r in res_416 if name in r['name']), 0)
        m640 = next((r['map50'] for r in res_640 if name in r['name']), 0)
        map_416_list.append(m416)
        map_640_list.append(m640)

    bars1 = ax.bar(x - width / 2, map_416_list, width, label='416x416',
                   color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width / 2, map_640_list, width, label='640x640',
                   color='#e74c3c', edgecolor='black')

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('mAP@0.5', fontsize=12)
    ax.set_title('Impact of Input Resolution', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([n.split()[-1] for n in model_names], rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存至 {save_path}")


def visualize_detections(image, boxes, scores, labels, class_names,
                         save_path=None, show=True):
    """
    在图像上绘制检测结果
    """
    img = image.copy()
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names))).tolist()

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        color = tuple([int(c * 255) for c in colors[int(label)][:3]])

        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 画标签
        label_text = f"{class_names[int(label)]}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(img, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if save_path:
        cv2.imwrite(save_path, img)
    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    return img


def plot_training_curves(history, save_path='training_curves.png'):
    """
    绘制训练过程中的损失和mAP曲线
    history: dict with keys 'train_loss', 'val_map', 'epochs'
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    ax = axes[0]
    for key in ['train_loss', 'val_loss']:
        if key in history:
            ax.plot(history['epochs'], history[key], label=key, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # mAP曲线
    ax = axes[1]
    for key in ['val_map50', 'val_map50_95']:
        if key in history:
            ax.plot(history['epochs'], history[key], label=key, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.set_title('Validation mAP')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_feature_maps(features, save_path='feature_maps.png', max_channels=16):
    """
    可视化中间层特征图
    features: [C, H, W] 或 [B, C, H, W]
    """
    if features.dim() == 4:
        features = features[0]  # 取第一个batch

    C = min(features.shape[0], max_channels)
    rows = int(np.ceil(np.sqrt(C)))
    cols = int(np.ceil(C / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if C > 1 else [axes]

    for i in range(C):
        feat = features[i].cpu().numpy()
        axes[i].imshow(feat, cmap='viridis')
        axes[i].set_title(f'Ch {i}')
        axes[i].axis('off')

    # 隐藏多余的子图
    for i in range(C, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pruning_sensitivity(analysis_results, save_path='pruning_sensitivity.png'):
    """
    绘制剪枝敏感度分析图
    analysis_results: list of dicts with 'ratio', 'map', 'params', 'flops'
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ratios = [r['ratio'] for r in analysis_results]
    maps = [r['map'] for r in analysis_results]
    params = [r['params'] for r in analysis_results]
    flops = [r['flops'] for r in analysis_results]

    # mAP vs 剪枝比例
    ax = axes[0]
    ax.plot(ratios, maps, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Pruning Ratio')
    ax.set_ylabel('mAP@0.5')
    ax.set_title('Accuracy vs Pruning Ratio')
    ax.grid(True, alpha=0.3)

    # 参数量 vs 剪枝比例
    ax = axes[1]
    ax.plot(ratios, params, 'go-', linewidth=2, markersize=8)
    ax.set_xlabel('Pruning Ratio')
    ax.set_ylabel('Parameters (M)')
    ax.set_title('Model Size vs Pruning Ratio')
    ax.grid(True, alpha=0.3)

    # FLOPs vs 剪枝比例
    ax = axes[2]
    ax.plot(ratios, flops, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Pruning Ratio')
    ax.set_ylabel('FLOPs (G)')
    ax.set_title('Computational Cost vs Pruning Ratio')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()