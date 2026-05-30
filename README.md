# 🔬 Medical Segmentation Mobile

Status: 🏗️ WIP - Phase 2 (Optimization & Edge Deployment)

Minimal, clean codebase for training 2D/3D UNet variants for medical image segmentation, with a focus on respecting memory constraints for edge/mobile inference. 

Include a LLM-based report generation to summarize the findings obtained by the segmentation model.


## 🗺️ Project Roadmap

| Phase | Milestone | Status |
| :--- | :--- | :--- |
| **Phase 1** | Core Training Pipeline (2.5D/3D U-Net, Deep Supervision) | ✅ Complete |
| **Phase 2** | **Edge Optimization (PTQ, ONNX Export & Benchmarking)** | 🚧 **In Progress** |
| **Phase 3** | **LLM Integration (Automated Findings Report Generation)** | 🚧 **In Progress** |

## 📁 Project Structure

```
src/
├── config.py         # nanoGPT-style configuration system
├── model.py          # UNet variants (2D/3D) with deep supervision
├── dataset.py        # Medical data loading (axial/coronal/sagittal/fullres)
├── transforms.py     # Medical augmentations (spatial/intensity)
├── loss.py           # Dice + CrossEntropy losses
├── optim.py          # Optimizers (AdamW/SGD) with proper parameter grouping
├── train.py          # Training loop 
├── eval.py           # 2.5D Inference & evaluation 
├── export.py         # ONNX export for mobile deployment
├── quantize.py       # Post-training quantization utilities
├── report.py         # Medical report generation
└── utils.py          # Memory reporting and utilities

ressources/ # 
├── images
└── templates


scripts/
└── report.py # LLM based report generation. Independant from rest of the code.  

data/ # MRI datasets
├── processed
└── raw

config/               # Training configurations
├── 2d_axi.py         # 2D axial slices (mobile-optimized)
├── 2d_cor.py         # 2D coronal slices  
├── 2d_sag.py         # 2D sagittal slices
└── 3d_fullres.py     # 3D full resolution (high accuracy)
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and setup
git clone <repository>
cd on-device-medical-seg

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Process raw medical data to expected format
python scripts/prepare.py --raw_dir ./BraTS2021 --save_dir ./data/processed
```

### 3. Train Models

```bash
# Complete training pipeline
bash train.sh

# Or specific configs
bash train.sh config/2d_axi.py    # Mobile-optimized 2D
bash train.sh config/3d_fullres.py # High-accuracy 3D

# training parameters can be overriden in the terminal as follow
python src/train.py config/2d_axi.py --batch_size=4 --learning_rate=0.001
```

## 📱 Optimization & Edge Deployment (WIP)

```python
# Export trained model
from src.export import export_to_onnx
export_to_onnx(model, "model.onnx", dynamic_axes=True)

# Quantize for mobile
from src.quantize import prepare_ptq, calibrate_ptq, finalize_ptq
quantized_model = prepare_ptq(model, backend="qnnpack")
# ... calibration ...
final_model = finalize_ptq(quantized_model)
```

## 🏥 Medical Reports (WIP)

Generate human-readable segmentation summaries:

```python
from src.report import generate_comprehensive_report

report = generate_comprehensive_report(
    predictions=model_output,
    class_names=["background", "tumor", "edema"],
    voxel_spacing=(1.0, 1.0, 1.0)
)
print(report)
```

## 🔧 Development

```bash
# Test individual components
python src/model.py     # Test model architecture
python src/dataset.py   # Test data loading
python src/transforms.py # Test augmentations

# Run evaluation
python src/eval.py --checkpoint checkpoints/best_model.pth --test_dir ./data/test
```

## 📜 License

MIT License (see LICENSE.md)

## 🙏 Acknowledgements  

- [karpathy/nanochat](https://github.com/karpathy/nanochat) - Design philosophy inspiration
- [nnUNet](https://github.com/MIC-DKFZ/nnUNet) - Medical segmentation insights
- Medical imaging community datasets
