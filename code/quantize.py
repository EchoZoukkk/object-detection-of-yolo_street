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
基于已有训练成果的INT8量化
无需重新训练，直接加载 best_640.pt / best_416.pt 等进行量化
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO
import time
import argparse
import json
import numpy as np
import yaml        # ← 修复1：添加yaml导入
import cv2         # ← 修复2：添加cv2导入
import glob

# ============ 配置 ============

# 你的已有模型路径（根据实际修改）
DEFAULT_MODELS = {
    'baseline_640': 'best_640.pt',
    'baseline_416': 'best_416.pt',
    'distilled_640': 'best_distill_640.pt',
    'distilled_416': 'best_distill_416.pt',  # ← 修复3：补全.pt后缀
    'pruned_30_640': 'yolov8n_pruned_30_finetune_640/weights/best.pt',
    'pruned_30_416': 'yolov8n_pruned_30_finetune_416/weights/best.pt',
}

SAVE_DIR = 'runs/quantize'


# ============ 图像预处理工具（新增） ============

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """将图像调整为模型输入尺寸，保持长宽比"""
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img


def preprocess_yolo(img_path, imgsz=640):
    """YOLO标准预处理"""
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
        if img is None:
            return None
    else:
        img = img_path.copy()

    # Letterbox resize
    img = letterbox(img, (imgsz, imgsz), auto=False)

    # BGR -> RGB, HWC -> CHW, 归一化
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    return img


# ============ 工具函数 ============

def get_model_size_mb(path):
    """获取模型文件大小(MB)"""
    return Path(path).stat().st_size / (1024 * 1024)


def load_model_safe(model_path, device='cpu'):
    """安全加载YOLO模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型不存在: {model_path}")

    try:
        model = YOLO(model_path)
        model.to(device)
        return model
    except Exception as e1:
        print(f"  标准加载失败: {e1}")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if 'model' in checkpoint:
                model = checkpoint['model']
            else:
                model = YOLO('yolov8n.pt')
                model.model.load_state_dict(checkpoint)
            model.to(device)
            return model
        except Exception as e2:
            raise RuntimeError(f"无法加载模型: {e2}")


# ============ PyTorch PTQ 静态量化 ============

class YOLO_PTQ_Quantizer:
    def __init__(self, model_path, device='cpu'):
        self.model_path = model_path
        self.device = device
        self.model_name = Path(model_path).stem

        print(f"\n{'='*60}")
        print(f"加载模型: {model_path}")
        print(f"{'='*60}")

        self.model = load_model_safe(model_path, device)
        self.model.model.eval()
        self.model.model.to('cpu')

        self.orig_size_mb = get_model_size_mb(model_path)
        print(f"原始模型大小: {self.orig_size_mb:.2f} MB")

    def fuse_modules(self):
        print("\n[1/4] 融合 Conv+BN 模块...")
        fused_count = 0

        def fuse_all_conv_bn(model, prefix=''):
            nonlocal fused_count
            for name, module in model.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if type(module).__name__ == 'Conv':
                    try:
                        if hasattr(module, 'conv') and hasattr(module, 'bn'):
                            torch.quantization.fuse_modules(module, ['conv', 'bn'], inplace=True)
                            fused_count += 1
                    except:
                        pass
                elif len(list(module.children())) > 0:
                    fuse_all_conv_bn(module, full_name)

        fuse_all_conv_bn(self.model.model)
        print(f"  融合模块数: {fused_count}")
        return self

    def prepare_for_ptq(self):
        print("\n[2/4] 准备PTQ量化配置...")
        self.model.model.qconfig = torch.quantization.get_default_qconfig('x86')
        torch.quantization.prepare(self.model.model, inplace=True)
        print("  已插入observer，等待校准数据...")
        return self

    def calibrate(self, data_yaml, num_images=100, imgsz=640):
        print(f"\n[3/4] 校准量化参数 ({num_images}张图像)...")

        # 从 data_yaml 解析验证集路径
        try:
            with open(data_yaml, 'r', encoding='utf-8') as f:
                data_cfg = yaml.safe_load(f)

            data_root = Path(data_cfg.get('path', '.'))
            if not data_root.is_absolute():
                yaml_dir = Path(data_yaml).parent.resolve()
                data_root = yaml_dir / data_root

            val_rel = data_cfg.get('val', 'val2017')
            val_path = data_root / val_rel

            print(f"  数据集根目录: {data_root}")
            print(f"  验证集路径: {val_path}")

            # 递归搜索所有子文件夹
            img_files = []
            for ext in ['jpg', 'jpeg', 'png', 'bmp', 'webp']:
                img_files.extend(val_path.rglob(f'*.{ext}'))
                img_files.extend(val_path.rglob(f'*.{ext.upper()}'))

            img_files = sorted(list(set([str(p) for p in img_files])))

            # 如果val2017为空，尝试train2017
            if len(img_files) == 0:
                print("  [!] val2017为空，尝试搜索 train2017...")
                train_path = data_root / 'train2017'
                for ext in ['jpg', 'jpeg', 'png', 'bmp', 'webp']:
                    img_files.extend(train_path.rglob(f'*.{ext}'))
                    img_files.extend(train_path.rglob(f'*.{ext.upper()}'))
                img_files = sorted(list(set([str(p) for p in img_files])))
                print(f"  从train2017找到 {len(img_files)} 张校准图像")
            else:
                print(f"  从val2017找到 {len(img_files)} 张校准图像")

            img_files = img_files[:num_images]

        except Exception as e:
            print(f"  [!] 解析 yaml 失败: {e}")
            img_files = []

        # 执行校准
        self.model.model.eval()

        if len(img_files) > 0:
            with torch.no_grad():
                for i, img_path in enumerate(img_files):
                    img_tensor = preprocess_yolo(img_path, imgsz)
                    if img_tensor is None:
                        continue

                    tensor = torch.from_numpy(img_tensor)
                    _ = self.model.model(tensor)

                    if (i + 1) % 20 == 0 or (i + 1) == len(img_files):
                        print(f"  校准进度: {i + 1}/{len(img_files)}")
            print("  校准完成！")

        else:
            print("  [!] 未找到校准图像，使用随机数据...")
            dummy = torch.randn(1, 3, imgsz, imgsz)
            with torch.no_grad():
                for i in range(num_images):
                    _ = self.model.model(dummy)
                    if (i + 1) % 20 == 0:
                        print(f"  随机校准进度: {i + 1}/{num_images}")

        return self

    def convert_and_save(self, save_dir=SAVE_DIR):
        print("\n[4/4] 转换为INT8模型...")
        int8_model = torch.quantization.convert(self.model.model, inplace=False)

        os.makedirs(save_dir, exist_ok=True)
        save_name = f"{self.model_name}_int8.pt"
        save_path = os.path.join(save_dir, save_name)

        torch.save({
            'model': int8_model,
            'state_dict': int8_model.state_dict(),
            'model_name': self.model_name,
            'quantize_type': 'ptq_int8',
        }, save_path)

        int8_size_mb = get_model_size_mb(save_path)
        compression_ratio = self.orig_size_mb / int8_size_mb if int8_size_mb > 0 else 0

        print(f"\n{'='*60}")
        print(f"PTQ量化完成!")
        print(f"  原始模型: {self.orig_size_mb:.2f} MB")
        print(f"  INT8模型: {int8_size_mb:.2f} MB")
        print(f"  压缩比:   {compression_ratio:.2f}x")
        print(f"  保存路径: {save_path}")
        print(f"{'='*60}")

        return save_path, compression_ratio


# ============ TensorRT / OpenVINO 导出（省略，与之前相同） ============

class TensorRT_INT8_Exporter:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_name = Path(model_path).stem

    def export_onnx(self, imgsz=640, simplify=True):
        print(f"\n{'='*60}")
        print(f"导出ONNX: {self.model_name}")
        print(f"{'='*60}")
        model = YOLO(self.model_path)
        onnx_path = self.model_path.replace('.pt', '.onnx')
        model.export(format='onnx', imgsz=imgsz, simplify=simplify, opset=12, dynamic=False)
        print(f"ONNX导出完成: {onnx_path}")
        return onnx_path

    def build_tensorrt_engine(self, onnx_path, imgsz=640, workspace_mb=4096, save_dir=SAVE_DIR):
        print(f"\n{'='*60}")
        print(f"构建TensorRT INT8引擎")
        print(f"{'='*60}")
        os.makedirs(save_dir, exist_ok=True)
        engine_path = os.path.join(save_dir, f"{self.model_name}_tensorrt_int8.engine")

        trtexec_cmd = (
            f"trtexec --onnx={onnx_path} --saveEngine={engine_path} --int8 "
            f"--workspace={workspace_mb} "
            f"--minShapes=images:1x3x{imgsz}x{imgsz} "
            f"--optShapes=images:1x3x{imgsz}x{imgsz} "
            f"--maxShapes=images:8x3x{imgsz}x{imgsz} --fp16 --verbose"
        )
        print(f"\n执行命令:\n{trtexec_cmd}")

        import subprocess
        result = subprocess.run(trtexec_cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"\nTensorRT引擎构建成功!")
            print(f"  保存路径: {engine_path}")
            orig_size = get_model_size_mb(self.model_path)
            engine_size = get_model_size_mb(engine_path)
            ratio = orig_size / engine_size if engine_size > 0 else 0
            print(f"  原始模型: {orig_size:.2f} MB")
            print(f"  INT8引擎: {engine_size:.2f} MB")
            print(f"  压缩比:   {ratio:.2f}x")
            return engine_path
        else:
            print(f"\n[!] TensorRT构建失败")
            print(f"错误输出:\n{result.stderr[:500]}")
            return None

    def export_full(self, imgsz=640, workspace_mb=4096, save_dir=SAVE_DIR):
        onnx_path = self.export_onnx(imgsz)
        engine_path = self.build_tensorrt_engine(onnx_path, imgsz, workspace_mb, save_dir)
        return engine_path


class OpenVINO_INT8_Exporter:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_name = Path(model_path).stem

    def export(self, imgsz=640, save_dir=SAVE_DIR):
        print(f"\n{'='*60}")
        print(f"OpenVINO INT8导出: {self.model_name}")
        print(f"{'='*60}")
        try:
            from openvino.tools.mo import convert_model
            from openvino.runtime import Core, serialize
        except ImportError:
            print("[!] OpenVINO未安装")
            return None

        model = YOLO(self.model_path)
        model.export(format='onnx', imgsz=imgsz, simplify=True)
        onnx_path = self.model_path.replace('.pt', '.onnx')

        print("转换ONNX -> OpenVINO IR...")
        ov_model = convert_model(onnx_path)

        os.makedirs(save_dir, exist_ok=True)
        xml_path = os.path.join(save_dir, f"{self.model_name}_openvino_int8.xml")
        serialize(ov_model, xml_path)
        print(f"OpenVINO模型保存: {xml_path}")
        return xml_path


# ============ 主函数 ============

def quantize_single_model(model_path, method='ptq', data_yaml='coco_street.yaml',
                          imgsz=640, num_calib=100, save_dir=SAVE_DIR):
    print(f"\n{'#'*70}")
    print(f"# 量化模型: {Path(model_path).name}")
    print(f"# 方法: {method}")
    print(f"# 分辨率: {imgsz}")
    print(f"{'#'*70}")

    results = {}

    if method in ['ptq', 'all']:
        print(f"\n{'-'*60}")
        print("执行 PyTorch PTQ 量化")
        print(f"{'-'*60}")
        try:
            quantizer = YOLO_PTQ_Quantizer(model_path)
            quantizer.fuse_modules()
            quantizer.prepare_for_ptq()
            quantizer.calibrate(data_yaml, num_calib, imgsz)
            ptq_path, ratio = quantizer.convert_and_save(save_dir)
            results['ptq'] = {'path': ptq_path, 'ratio': ratio}
        except Exception as e:
            print(f"[!] PTQ量化失败: {e}")
            import traceback
            traceback.print_exc()

    if method in ['tensorrt', 'all']:
        print(f"\n{'-'*60}")
        print("执行 TensorRT INT8 导出")
        print(f"{'-'*60}")
        try:
            exporter = TensorRT_INT8_Exporter(model_path)
            engine_path = exporter.export_full(imgsz, save_dir=save_dir)
            if engine_path:
                results['tensorrt'] = {'path': engine_path}
        except Exception as e:
            print(f"[!] TensorRT导出失败: {e}")

    if method in ['openvino', 'all']:
        print(f"\n{'-'*60}")
        print("执行 OpenVINO INT8 导出")
        print(f"{'-'*60}")
        try:
            exporter = OpenVINO_INT8_Exporter(model_path)
            ov_path = exporter.export(imgsz, save_dir)
            if ov_path:
                results['openvino'] = {'path': ov_path}
        except Exception as e:
            print(f"[!] OpenVINO导出失败: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description='基于已有模型的INT8量化（无需重新训练）')
    parser.add_argument('--model', type=str, required=True,
                        help='已有模型路径，如 best_640.pt 或 runs/distill/best_distill.pt')
    parser.add_argument('--data', type=str, default='coco_street.yaml',
                        help='数据集配置文件')
    parser.add_argument('--method', type=str, default='ptq',
                        choices=['ptq', 'tensorrt', 'openvino', 'all'],
                        help='量化方法')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入分辨率')
    parser.add_argument('--num-calib', type=int, default=100,
                        help='PTQ校准图像数量')
    parser.add_argument('--save-dir', type=str, default='runs/quantize',
                        help='保存目录')
    parser.add_argument('--batch-quantize', action='store_true',
                        help='批量量化所有已有模型')

    args = parser.parse_args()

    if args.batch_quantize:
        print("批量量化模式：量化所有已有模型")
        all_models = []
        for name, path in DEFAULT_MODELS.items():
            if os.path.exists(path):
                all_models.append((name, path))

        if not all_models:
            print("[!] 未找到任何已有模型，请检查路径")
            return

        print(f"找到 {len(all_models)} 个模型:")
        for name, path in all_models:
            print(f"  {name}: {path}")

        all_results = {}
        for name, path in all_models:
            imgsz = 416 if '416' in name else 640
            result = quantize_single_model(path, args.method, args.data, imgsz,
                                            args.num_calib, args.save_dir)
            all_results[name] = result

        summary_path = os.path.join(args.save_dir, 'quantization_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n汇总结果保存: {summary_path}")

    else:
        quantize_single_model(args.model, args.method, args.data,
                              args.imgsz, args.num_calib, args.save_dir)


if __name__ == '__main__':
    main()