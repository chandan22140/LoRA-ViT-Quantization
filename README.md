# LoRA-ViT-Quantization

## Project Description
LoRA-ViT-Quantization is a PyTorch-based deep learning project that fine-tunes Vision Transformer (ViT-B/16) using LoRA (Low-Rank Adaptation) while leveraging BitsandBytes quantization for efficient training. The implementation reduces the number of trainable parameters by modifying the linear layers within the ViT model, making it more memory efficient while preserving accuracy. The project is designed to train on Tiny ImageNet and supports multi-GPU training with TensorBoard logging.

## Features
- Fine-tunes ViT-B/16 using LoRA to reduce memory footprint and improve efficiency.
- Uses BitsandBytes quantization (NF4) for lower precision weight storage.
- Implements trainable parameter filtering and optimizer setup for fine-tuning.
- Supports multi-GPU training.
- Integrates TensorBoard for logging training loss and validation accuracy.

## Installation
### Requirements
- Python 3.8+
- PyTorch
- torchvision
- bitsandbytes
- TensorBoard
- matplotlib

### Install Dependencies
```bash
pip install torch torchvision bitsandbytes tensorboard matplotlib
```

## Dataset
This project uses **Tiny ImageNet**. Ensure that the dataset is placed correctly before training:
```bash
/kaggle/input/tiny-imagenet/tiny-imagenet-200/
```

## Training
To train the model, run:
```bash
python train.py  # Modify if needed
```
The script includes:
- Data preprocessing (Resizing, Tensor conversion)
- Replacing linear layers with LoRA-wrapped quantized layers
- Marking only LoRA and classification head as trainable
- Training loop with AdamW optimizer and learning rate scheduling
- TensorBoard logging

## Model Modifications
- Replaces `nn.Linear` layers with `QuantizedLoRALinear` using BitsandBytes.
- Freezes non-LoRA parameters to optimize fine-tuning.
- Uses dropout in the classification head for regularization.

## Results and Logging
Training progress and performance metrics can be viewed using TensorBoard:
```bash
tensorboard --logdir=tensorboard_logs/
```

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
MIT License

