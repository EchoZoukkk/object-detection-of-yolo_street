# Ultra-Lightweight YOLOv8 Street Scene Object Detection
Team practical project: Lightweight YOLOv8 detection for street traffic scenarios, based on YOLOv8n, optimized via knowledge distillation, channel pruning and INT8 quantization.

## Project Overview
This project takes YOLOv8n as the baseline model for street object detection. Target categories include pedestrians, vehicles, traffic lights, traffic signs, motorcycles and bicycles.
To achieve edge-deployable ultra-lightweight model, we adopt three core compression strategies:
1. Knowledge Distillation: Take YOLOv8m as teacher model to transfer feature and classification knowledge to YOLOv8n student model.
2. Channel Pruning: L1-norm based structured pruning to remove redundant convolution channels, reduce parameters and FLOPs.
3. INT8 Quantization: Post-training quantization to further reduce model size and speed up inference on embedded devices.

## Directory Structure
object-detection-of-yolo_street/
├── LICENSE # Apache 2.0 Open Source License
├── coco_street.yaml # COCO street scene dataset config
├── train_baseline.py # Train original YOLOv8n baseline model
├── train_distillation.py # Two-stage knowledge distillation training
├── train_pruning.py # Model pruning + post-finetune pipeline
├── quantize.py # INT8 quantization script
├── convert-coco2yolo.py # Dataset format conversion tool
├── evaluate.py # Model evaluation script
├── models/
│ ├── init.py
│ ├── distillation.py # Distillation loss & feature extractor module
│ └── pruning.py # Manual structured pruning implementation
└── utils/
├── init.py
├── dataset.py # COCO street dataset filtering & augmentation
├── metrics.py # Custom mAP & IoU calculation
└── visualize.py # Training curve & detection result visualization

## Environment & Dependencies
Create Python environment and install required packages:
```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows Git Bash / CMD

# Install dependencies
pip install torch torchvision ultralytics opencv-python pycocotools matplotlib numpy
Core version reference:
Python >= 3.9
PyTorch >= 2.0
Ultralytics YOLOv8 >= 8.0
CUDA (optional, for GPU training acceleration)

How to Run
1. Dataset Preparation
Download COCO 2017 train/val dataset
Modify path field in coco_street.yaml to your local dataset root path
Run dataset filter script to extract street-related objects:
python convert-coco2yolo.py
2. Train Baseline YOLOv8n Model
python train_baseline.py --data coco_street.yaml --epochs 50 --imgsz 640 --batch 16
3. Knowledge Distillation Training
python train_distillation.py --data coco_street.yaml --epochs 50
4. Model Channel Pruning
python train_pruning.py --model runs/train2017/yolov8n_street_640/weights/best.pt --ratio 0.3
5. INT8 Quantization
python quantize.py
6. Model Evaluation & Visualization
python evaluate.py

Model Optimization Effect
Baseline YOLOv8n: Original parameters, high inference latency
Distilled YOLOv8n: Accuracy improved, no extra computation cost
Pruned YOLOv8n: Parameters & FLOPs reduced significantly
Quantized YOLOv8n: Final ultra-lightweight model for edge deployment

License
This project is licensed under the Apache License 2.0 - see the LICENSE file for full details.
