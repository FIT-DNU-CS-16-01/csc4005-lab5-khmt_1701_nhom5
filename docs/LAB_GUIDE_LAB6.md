# File: docs/LAB_GUIDE_LAB6.md

```md
# LAB GUIDE LAB 6

## Vision Transformer Indoor Scene Classification

### Mục tiêu

Xây dựng hệ thống phân loại cảnh trong nhà sử dụng Vision Transformer (ViT).

### Dataset

MIT Indoor Scene Recognition Dataset.

Các lớp sử dụng:

- classroom
- computerroom
- library
- corridor
- office

### Mô hình

- ViT-B/16 pretrained
- Head-only training
- Full fine-tuning

### Công nghệ sử dụng

- Python
- PyTorch
- torchvision
- wandb

### Kết quả đạt được

- Test Accuracy ≈ 92%
- Macro-F1 ≈ 91%

### Weights & Biases

https://wandb.ai/ducduonglop8a-hanoi/csc4005-lab6-mit-indoor-vit?nw=nwuserducduonglop8a