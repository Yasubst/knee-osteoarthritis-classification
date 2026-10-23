
# ==================================================
# KNEE OSTEOARTHRITIS CLASSIFICATION
# PRETRAINED RESNET18 CNN MODEL
# ==================================================

import torch
import torch.nn as nn
import torchvision.models as models


def create_model(num_classes=5):

    # Load ResNet18 with pretrained ImageNet weights
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Original convolution layer accepts 3 RGB channels
    old_conv = model.conv1

    # New convolution layer accepts 1 grayscale channel
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    # Convert pretrained RGB weights to grayscale weights
    with torch.no_grad():
        new_conv.weight.copy_(
            old_conv.weight.mean(dim=1, keepdim=True)
        )

    # Replace the first convolution layer
    model.conv1 = new_conv

    # Replace final classification layer for 5 KL grades
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


# ==================================================
# TEST MODEL ARCHITECTURE
# ==================================================

if __name__ == "__main__":

    model = create_model(num_classes=5)

    # Create 2 dummy grayscale images
    dummy_images = torch.randn(2, 1, 224, 224)

    model.eval()

    with torch.no_grad():
        outputs = model(dummy_images)

    print("----- Pretrained CNN Model Test -----")
    print("Input shape:", dummy_images.shape)
    print("Output shape:", outputs.shape)
    print("Input channels:", model.conv1.in_channels)
    print("Output classes:", model.fc.out_features)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Total model parameters:", total_parameters)
