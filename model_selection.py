import segmentation_models_pytorch as smp
import numpy as np
from torchvision import models
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define label map and labels (for visualization purposes)
label_map = np.array([
    (0, 0, 0),          # 0 - Background (Black)
    (0, 0, 255),        # 1 - Surface water (Blue)
    (135, 206, 250),    # 2 - Street (Light Sky Blue)
    (255, 255, 0),      # 3 - Urban Fabric (Yellow)
    (128, 0, 0),        # 4 - Industrial, commercial and transport (Maroon)
    (139, 37, 0),       # 5 - Mine, dump, and construction sites (Reddish Brown)
    (0, 128, 0),        # 6 - Artificial, vegetated areas (Green)
    (255, 165, 0),      # 7 - Arable Land (Orange)
    (0, 255, 0),        # 8 - Permanent Crops (Lime Green)
    (154, 205, 50),     # 9 - Pastures (Yellow Green)
    (34, 139, 34),      # 10 - Forests (Forest Green)
    (139, 69, 19),      # 11 - Shrub (Saddle Brown)
    (245, 245, 220),    # 12 - Open spaces with no vegetation (Beige)
    (0, 255, 255),      # 13 - Inland wetlands (Cyan)
])

labels = [
    "Background", "Surface water", "Street", "Urban Fabric", "Industrial, commercial and transport",
    "Mine, dump, and construction sites", "Artificial, vegetated areas", "Arable Land",
    "Permanent Crops", "Pastures", "Forests", "Shrub", "Open spaces with no vegetation", "Inland wetlands"
]

model_unetPlusPlus = smp.UnetPlusPlus(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=6,
    classes=len(label_map) 
)

class CustomDeepLabV3(nn.Module):
    def __init__(self, num_classes):
        super(CustomDeepLabV3, self).__init__()
        self.model = smp.DeepLabV3(
            encoder_name="resnet50",       
            encoder_weights="imagenet",  
            classes=num_classes,          
            activation=None               
        )
        
        self.model.encoder.conv1 = nn.Conv2d(
            in_channels=6, 
            out_channels=self.model.encoder.conv1.out_channels,
            kernel_size=self.model.encoder.conv1.kernel_size,
            stride=self.model.encoder.conv1.stride,
            padding=self.model.encoder.conv1.padding,
            bias=False
        )

    def forward(self, x):
        return self.model(x)

num_classes = 14  
model = CustomDeepLabV3(num_classes=num_classes).to(device)

model_linknet = smp.Linknet(encoder_name='resnet34',
                                          encoder_depth=5,
                                          encoder_weights='imagenet',
                                          decoder_use_batchnorm=True,
                                          in_channels=6,
                                          classes=14,
                                          activation=None,
                                          aux_params=None)