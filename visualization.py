import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

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

# Function to convert label image to RGB for visualization
def label_to_rgb(label_image, label_map):
    rgb_image = np.zeros((*label_image.shape, 3), dtype=np.uint8)
    for label, color in enumerate(label_map):
        rgb_image[label_image == label] = color
    return rgb_image

def visualize_predictions(model, dataset, label_map, labels, device, num_samples=10):
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)
    samples = [dataset[i] for i in indices]
    
    fig, axs = plt.subplots(num_samples, 2, figsize=(12, 5 * num_samples))
    
    for i, (image, true_label) in enumerate(samples):
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
        pred_label = torch.argmax(output, dim=1).cpu().numpy().squeeze()

        true_label = true_label.numpy()
        
        axs[i, 0].imshow(label_to_rgb(true_label, label_map))
        axs[i, 0].set_title(f"Sample {indices[i]} - Ground Truth")
        
        axs[i, 1].imshow(label_to_rgb(pred_label, label_map))
        axs[i, 1].set_title(f"Sample {indices[i]} - Prediction")
        
        for ax in axs[i]:
            ax.axis('off')
    
    # Convert label_map colors to tuple format (0-1 range) for legend
    legend_handles = [plt.Line2D([0], [0], color=color/255, lw=4) for color in label_map]
    
    # Legend
    fig.legend(handles=legend_handles, labels=labels, loc='upper right')
    
    plt.tight_layout()
    plt.show()
