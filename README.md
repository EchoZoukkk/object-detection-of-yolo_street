# Ultra-Lightweight YOLOv8 Street Scene Object Detection
Team practical project: Lightweight YOLOv8 detection for street traffic scenarios, based on YOLOv8n, optimized via knowledge distillation, channel pruning and INT8 quantization.

## Project Overview
This project takes YOLOv8n as the baseline model for street object detection. Target categories include pedestrians, vehicles, traffic lights, traffic signs, motorcycles and bicycles.
To achieve edge-deployable ultra-lightweight model, we adopt three core compression strategies:
1. Knowledge Distillation: Take YOLOv8m as teacher model to transfer feature and classification knowledge to YOLOv8n student model.
2. Channel Pruning: L1-norm based structured pruning to remove redundant convolution channels, reduce parameters and FLOPs.
3. INT8 Quantization: Post-training quantization to further reduce model size and speed up inference on embedded devices.

## Directory Structure
