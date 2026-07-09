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
模型评估：FLOPs、参数量、mAP、推理速度
支持模型：Baseline (416/640) + Pruned (416/640)
自动适配 CPU/GPU 环境
"""

import torch
import time
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
import matplotlib.pyplot as plt
from thop import profile, clever_format
import warnings
import pandas as pd

warnings.filterwarnings('ignore')


# ==================== 自动检测设备 ====================
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"检测到设备: {DEVICE}")
if DEVICE == 'cpu':
    print("⚠️  未检测到 CUDA GPU，将使用 CPU 进行 mAP 计算（速度较慢）")


def measure_model_complexity(model, img_size=640):
    """
    计算模型参数量和FLOPs
    """
    device = next(model.parameters()).device
    example_input = torch.randn(1, 3, img_size, img_size).to(device)

    # 使用thop计算FLOPs
    flops, params = profile(model, inputs=(example_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    # 同时计算原始数值用于绘图
    params_raw = sum(p.numel() for p in model.parameters())
    flops_raw = float(flops.replace('M', '').replace('G', '').replace('K', ''))
    if 'G' in flops:
        flops_raw *= 1e9
    elif 'M' in flops:
        flops_raw *= 1e6
    elif 'K' in flops:
        flops_raw *= 1e3

    return {
        'params': params_raw,
        'params_M': params_raw / 1e6,
        'flops': flops_raw,
        'flops_G': flops_raw / 1e9,
        'params_str': params,
        'flops_str': flops
    }


def measure_inference_speed(model, img_size=640, warmup=50, repeats=200):
    """
    测量纯网络推理速度（不含预处理/后处理）
    """
    device = next(model.parameters()).device
    example_input = torch.randn(1, 3, img_size, img_size).to(device)
    model.eval()

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(example_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 正式测试
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(example_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    return {
        'avg_latency_ms': avg_time * 1000,
        'fps': fps,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000
    }


def evaluate_single_model(model_path, data_yaml, img_size=640, name='model'):
    """
    评估单个模型
    """
    print(f"\n{'=' * 70}")
    print(f"评估模型: {name} @ {img_size}x{img_size}")
    print(f"{'=' * 70}")

    # 加载模型
    model = YOLO(model_path)

    # 获取PyTorch模型用于复杂度分析
    pt_model = model.model

    print(f"\n[1/4] 验证模型路径: {model_path}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    print(f"\n[2/4] 计算 mAP... (设备: {DEVICE})")
    try:
        # 关键修复：使用自动检测的设备，不再硬编码 device=0
        metrics = model.val(data=data_yaml, imgsz=img_size, verbose=False, device=DEVICE)
        map50 = metrics.box.map50
        map50_95 = metrics.box.map
        print(f"  mAP@0.5:      {map50:.4f}")
        print(f"  mAP@0.5:0.95: {map50_95:.4f}")
    except Exception as e:
        print(f"  ❌ mAP计算失败: {e}")
        map50 = 0.0
        map50_95 = 0.0

    print(f"\n[3/4] 分析模型复杂度...")
    try:
        complexity = measure_model_complexity(pt_model, img_size)
        print(f"  参数量: {complexity['params_M']:.2f}M ({complexity['params_str']})")
        print(f"  FLOPs:  {complexity['flops_G']:.2f}G ({complexity['flops_str']})")
    except Exception as e:
        print(f"  ❌ 复杂度分析失败: {e}")
        complexity = {'params_M': 0, 'flops_G': 0, 'params': 0, 'flops': 0}

    print(f"\n[4/4] 测量推理速度 (warmup=50, repeats=200)...")
    try:
        speed = measure_inference_speed(pt_model, img_size)
        print(f"  平均延迟: {speed['avg_latency_ms']:.2f}ms ± {speed['std_ms']:.2f}ms")
        print(f"  FPS:      {speed['fps']:.2f}")
        print(f"  延迟范围: [{speed['min_ms']:.2f}, {speed['max_ms']:.2f}] ms")
    except Exception as e:
        print(f"  ❌ 速度测试失败: {e}")
        speed = {'avg_latency_ms': 0, 'fps': 0, 'std_ms': 0}

    return {
        'name': name,
        'img_size': img_size,
        'map50': map50,
        'map50_95': map50_95,
        'params_M': complexity['params_M'],
        'flops_G': complexity['flops_G'],
        'params': complexity['params'],
        'flops': complexity['flops'],
        'latency_ms': speed['avg_latency_ms'],
        'fps': speed['fps'],
        'std_ms': speed['std_ms']
    }


def compare_models(data_yaml='coco_street.yaml', output_dir='evaluation_output'):
    """
    对比所有模型：基线(416/640) + 剪枝(416/640)
    每个模型只测试其训练时的分辨率
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # ==================== 模型配置 ====================
    # ⚠️ 请根据你的实际文件路径修改下面这4个路径！
    model_configs = [
        {
            'name': 'YOLOv8n-Baseline-640',
            'path': 'best_640.pt',           # ← 你的基线640模型
            'img_size': 640
        },
        {
            'name': 'YOLOv8n-Baseline-416',
            'path': 'best_416.pt',           # ← 你的基线416模型
            'img_size': 416
        },
        {
            'name': 'YOLOv8n-Pruned-640',
            'path': 'pruned_best640.pt',         # ← 改成你的剪枝640模型路径
            'img_size': 640
        },
        {
            'name': 'YOLOv8n-Pruned-416',
            'path': 'pruned_best416.pt',         # ← 改成你的剪枝416模型路径
            'img_size': 416
        },
    ]

    # 评估每个模型
    for config in model_configs:
        model_path = config['path']
        if not Path(model_path).exists():
            print(f"\n⚠️  跳过 {config['name']}: 模型不存在 ({model_path})")
            print(f"   请修改代码中的路径或确认文件已生成")
            continue

        try:
            result = evaluate_single_model(
                model_path=model_path,
                data_yaml=data_yaml,
                img_size=config['img_size'],
                name=config['name']
            )
            results.append(result)
        except Exception as e:
            print(f"\n❌ 评估失败 {config['name']}: {e}")

    # 保存JSON结果
    if results:
        json_path = output_dir / 'evaluation_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 结果已保存至: {json_path}")

        # 生成对比表格和图表
        print_comparison_table(results)
        plot_comparison(results, output_dir)
    else:
        print("\n❌ 没有成功评估任何模型，请检查路径配置")

    return results


def print_comparison_table(results):
    """
    打印对比表格
    """
    print(f"\n{'=' * 100}")
    print("模型性能对比表")
    print(f"{'=' * 100}")

    # 创建DataFrame
    df = pd.DataFrame(results)
    df = df[['name', 'img_size', 'map50', 'map50_95', 'params_M', 'flops_G', 'latency_ms', 'fps']]
    df.columns = ['模型', '分辨率', 'mAP@0.5', 'mAP@0.5:0.95', '参数量(M)', 'FLOPs(G)', '延迟(ms)', 'FPS']

    # 格式化输出
    print(df.to_string(index=False, float_format='%.4f'))
    print(f"{'=' * 100}\n")

    # 计算压缩比和加速比（以同分辨率基线为基准）
    print("压缩与加速分析（以同分辨率基线为基准）:")
    print("-" * 80)

    baselines = {}
    for r in results:
        if 'Baseline' in r['name']:
            res = r['img_size']
            baselines[res] = r

    for r in results:
        if 'Pruned' in r['name']:
            res = r['img_size']
            if res in baselines:
                base = baselines[res]
                params_ratio = r['params_M'] / base['params_M'] * 100
                flops_ratio = r['flops_G'] / base['flops_G'] * 100
                speedup = base['latency_ms'] / r['latency_ms'] if r['latency_ms'] > 0 else 0
                map_drop = (base['map50'] - r['map50']) * 100  # 百分点

                print(f"\n{r['name']} vs {base['name']}:")
                print(f"  参数量:  {r['params_M']:.2f}M / {base['params_M']:.2f}M = {params_ratio:.1f}%")
                print(f"  FLOPs:   {r['flops_G']:.2f}G / {base['flops_G']:.2f}G = {flops_ratio:.1f}%")
                if speedup > 0:
                    print(f"  加速比:  {base['latency_ms']:.2f}ms / {r['latency_ms']:.2f}ms = {speedup:.2f}x")
                else:
                    print(f"  加速比:  无法计算（剪枝模型延迟为0）")
                print(f"  mAP下降: {base['map50']:.4f} -> {r['map50']:.4f} = -{map_drop:.2f}%")


def plot_comparison(results, output_dir):
    """
    生成对比图表
    """
    if not results:
        return

    # 分离不同分辨率
    res_416 = [r for r in results if r['img_size'] == 416]
    res_640 = [r for r in results if r['img_size'] == 640]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('YOLOv8n Model Comparison: Baseline vs Pruned', fontsize=14, fontweight='bold')

    # 1. mAP vs FLOPs
    ax = axes[0, 0]
    for res, color, marker, label in [
        (res_416, '#3498db', 'o', '416x416'),
        (res_640, '#e74c3c', 's', '640x640')
    ]:
        if res:
            names = [r['name'].replace('YOLOv8n-', '').replace('-416', '').replace('-640', '') for r in res]
            maps = [r['map50'] for r in res]
            flops = [r['flops_G'] for r in res]
            ax.scatter(flops, maps, c=color, marker=marker, s=150, edgecolors='black', linewidth=1, label=label, zorder=5)
            for i, name in enumerate(names):
                ax.annotate(name, (flops[i], maps[i]), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('FLOPs (G)', fontsize=11)
    ax.set_ylabel('mAP@0.5', fontsize=11)
    ax.set_title('mAP vs FLOPs', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')

    # 2. mAP vs Parameters
    ax = axes[0, 1]
    for res, color, marker, label in [
        (res_416, '#3498db', 'o', '416x416'),
        (res_640, '#e74c3c', 's', '640x640')
    ]:
        if res:
            names = [r['name'].replace('YOLOv8n-', '').replace('-416', '').replace('-640', '') for r in res]
            maps = [r['map50'] for r in res]
            params = [r['params_M'] for r in res]
            ax.scatter(params, maps, c=color, marker=marker, s=150, edgecolors='black', linewidth=1, label=label, zorder=5)
            for i, name in enumerate(names):
                ax.annotate(name, (params[i], maps[i]), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('Parameters (M)', fontsize=11)
    ax.set_ylabel('mAP@0.5', fontsize=11)
    ax.set_title('mAP vs Parameters', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')

    # 3. mAP vs Latency
    ax = axes[0, 2]
    for res, color, marker, label in [
        (res_416, '#3498db', 'o', '416x416'),
        (res_640, '#e74c3c', 's', '640x640')
    ]:
        if res:
            names = [r['name'].replace('YOLOv8n-', '').replace('-416', '').replace('-640', '') for r in res]
            maps = [r['map50'] for r in res]
            latency = [r['latency_ms'] for r in res]
            ax.scatter(latency, maps, c=color, marker=marker, s=150, edgecolors='black', linewidth=1, label=label, zorder=5)
            for i, name in enumerate(names):
                ax.annotate(name, (latency[i], maps[i]), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('Latency (ms)', fontsize=11)
    ax.set_ylabel('mAP@0.5', fontsize=11)
    ax.set_title('Accuracy-Speed Trade-off', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')

    # 4. 分辨率对比 - mAP
    ax = axes[1, 0]
    model_types = ['Baseline', 'Pruned']
    x = np.arange(len(model_types))
    width = 0.35

    map_416 = []
    map_640 = []
    for mtype in model_types:
        m416 = next((r['map50'] for r in res_416 if mtype in r['name']), 0)
        m640 = next((r['map50'] for r in res_640 if mtype in r['name']), 0)
        map_416.append(m416)
        map_640.append(m640)

    bars1 = ax.bar(x - width/2, map_416, width, label='416x416', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, map_640, width, label='640x640', color='#e74c3c', edgecolor='black')
    ax.set_xlabel('Model Type', fontsize=11)
    ax.set_ylabel('mAP@0.5', fontsize=11)
    ax.set_title('Resolution Impact on mAP', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 5. 压缩效果对比 - 参数量
    ax = axes[1, 1]
    baseline_416_params = next((r['params_M'] for r in res_416 if 'Baseline' in r['name']), 0)
    pruned_416_params = next((r['params_M'] for r in res_416 if 'Pruned' in r['name']), 0)
    baseline_640_params = next((r['params_M'] for r in res_640 if 'Baseline' in r['name']), 0)
    pruned_640_params = next((r['params_M'] for r in res_640 if 'Pruned' in r['name']), 0)

    params_data = {
        '416x416': [baseline_416_params, pruned_416_params],
        '640x640': [baseline_640_params, pruned_640_params]
    }

    x = np.arange(len(model_types))
    width = 0.35
    colors = ['#3498db', '#e74c3c']

    for i, (res, vals) in enumerate(params_data.items()):
        bars = ax.bar(x + i*width - width/2, vals, width, label=res, color=colors[i], edgecolor='black')
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.1f}M', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model Type', fontsize=11)
    ax.set_ylabel('Parameters (M)', fontsize=11)
    ax.set_title('Parameter Compression', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    # 6. 推理速度对比
    ax = axes[1, 2]
    baseline_416_lat = next((r['latency_ms'] for r in res_416 if 'Baseline' in r['name']), 0)
    pruned_416_lat = next((r['latency_ms'] for r in res_416 if 'Pruned' in r['name']), 0)
    baseline_640_lat = next((r['latency_ms'] for r in res_640 if 'Baseline' in r['name']), 0)
    pruned_640_lat = next((r['latency_ms'] for r in res_640 if 'Pruned' in r['name']), 0)

    latency_data = {
        '416x416': [baseline_416_lat, pruned_416_lat],
        '640x640': [baseline_640_lat, pruned_640_lat]
    }

    for i, (res, vals) in enumerate(latency_data.items()):
        bars = ax.bar(x + i*width - width/2, vals, width, label=res, color=colors[i], edgecolor='black')
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}ms', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model Type', fontsize=11)
    ax.set_ylabel('Latency (ms)', fontsize=11)
    ax.set_title('Inference Latency', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_results.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ 对比图已保存至: {output_dir / 'comparison_results.png'}")
    plt.show()

    # 额外生成论文用图
    plot_paper_figure(results, output_dir)


def plot_paper_figure(results, output_dir):
    """
    生成适合论文插入的综合对比图
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {
        'Baseline-416': '#2ecc71',
        'Pruned-416': '#27ae60',
        'Baseline-640': '#3498db',
        'Pruned-640': '#2980b9'
    }

    markers = {
        'Baseline-416': 'o',
        'Pruned-416': 's',
        'Baseline-640': 'D',
        'Pruned-640': '^'
    }

    for r in results:
        key = r['name'].replace('YOLOv8n-', '')
        label = key.replace('-', ' @ ')
        ax.scatter(r['flops_G'], r['map50'],
                  c=colors.get(key, '#95a5a6'),
                  marker=markers.get(key, 'o'),
                  s=300, edgecolors='black', linewidth=2,
                  label=label, zorder=5)
        ax.annotate(f"{r['map50']:.3f}\n{r['latency_ms']:.1f}ms",
                   (r['flops_G'], r['map50']),
                   textcoords="offset points", xytext=(15, 0),
                   ha='left', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor=colors.get(key, '#95a5a6'), alpha=0.8))

    ax.set_xlabel('FLOPs (G)', fontsize=13, fontweight='bold')
    ax.set_ylabel('mAP@0.5', fontsize=13, fontweight='bold')
    ax.set_title('Model Comparison: Accuracy vs Complexity', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / 'paper_figure.png', dpi=300, bbox_inches='tight')
    print(f"✅ 论文用图已保存至: {output_dir / 'paper_figure.png'}")
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate YOLOv8n Baseline vs Pruned models')
    parser.add_argument('--data', type=str, default='coco_street.yaml', help='Path to data YAML file')
    parser.add_argument('--output', type=str, default='evaluation_output', help='Output directory for results')
    args = parser.parse_args()

    results = compare_models(args.data, args.output)

    print("\n" + "=" * 70)
    print("评估完成！所有结果保存在:", Path(args.output).absolute())
    print("=" * 70)