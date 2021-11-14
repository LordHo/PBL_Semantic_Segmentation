import segmentation_models_pytorch as smp
import torchsummary
import torch

model = smp.PSPNet(
    encoder_name='efficientnet-b7',
    in_channels=3,
    classes=11,
    activation='softmax2d'
)
input_size = (3, 512, 512)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
torchsummary.summary(model, input_size)