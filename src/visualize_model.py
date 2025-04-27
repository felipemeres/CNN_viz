import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from visualization_utils import load_image, register_hooks, run_forward, cleanup_hooks, visualize_activations
import os
from dotenv import load_dotenv
from PIL import Image
import numpy as np

# --------------------
# User Parameters (now loaded from environment variables)
# --------------------
load_dotenv()

model_type = os.getenv('MODEL_TYPE', 'alexnet')  # 'alexnet' or 'resnet18'
img_path = os.getenv('IMG_PATH', 'sample_image.jpg')
img_size = tuple(map(int, os.getenv('IMG_SIZE', '600,600').split(',')))  # Image resize dimensions
show_legend = os.getenv('SHOW_LEGEND', 'False').lower() in ('true', '1', 'yes')  # True to include text/legend, False for clean images
save_video = os.getenv('SAVE_VIDEO', 'True').lower() in ('true', '1', 'yes')  # True to save video
video_filename = os.getenv('VIDEO_FILENAME', 'activations.avi')  # Output video file (if saving video)
fps = int(os.getenv('FPS', '24'))  # Frames per second for the video
cmap = os.getenv('CMAP', 'bone')  # Colormap for visualization (e.g., 'viridis', 'gray', 'plasma', etc.)
normalization_mode = os.getenv('NORMALIZATION_MODE', 'filter')  # 'filter', 'layer', or 'global'
use_hist_eq = os.getenv('USE_HIST_EQ', 'False').lower() in ('true', '1', 'yes')  # True to use histogram equalization
clip_percentiles = tuple(map(int, os.getenv('CLIP_PERCENTILES', '1,99').split(',')))  # e.g., (1, 99) to clip outliers
save_16bit = os.getenv('SAVE_16BIT', 'False').lower() in ('true', '1', 'yes')  # True to save 16-bit PNGs
overlay_on_image = os.getenv('OVERLAY_ON_IMAGE', 'False').lower() in ('true', '1', 'yes')
show_raw_stats = os.getenv('SHOW_RAW_STATS', 'False').lower() in ('true', '1', 'yes')
show_moving_filter = os.getenv('SHOW_MOVING_FILTER', 'False').lower() in ('true', '1', 'yes')
# --------------------

# 1. Load and preprocess the image
input_tensor = load_image(img_path, img_size)

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Load the selected model and set up layer names and output directory
if model_type.lower() == 'alexnet':
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.eval()
    layer_names = [
        # --- Features (convolutional part) ---
        ('features_0', model.features[0]),  # Conv1
        ('features_1', model.features[1]),  # ReLU1
        ('features_2', model.features[2]),  # MaxPool1
        ('features_3', model.features[3]),  # Conv2
        ('features_4', model.features[4]),  # ReLU2
        ('features_5', model.features[5]),  # MaxPool2
        ('features_6', model.features[6]),  # Conv3
        ('features_7', model.features[7]),  # ReLU3
        ('features_8', model.features[8]),  # Conv4
        ('features_9', model.features[9]),  # ReLU4
        ('features_10', model.features[10]), # Conv5
        ('features_11', model.features[11]), # ReLU5
        ('features_12', model.features[12]), # MaxPool3
        # --- AdaptiveAvgPool2d ---
        ('avgpool', model.avgpool),
        # --- Classifier (fully connected part) ---
        # Dropout layers are skipped
        # ('classifier_1', model.classifier[1]),  # Linear1
        # ('classifier_2', model.classifier[2]),  # ReLU6
        # ('classifier_4', model.classifier[4]),  # Linear2
        # ('classifier_5', model.classifier[5]),  # ReLU7
        # ('classifier_6', model.classifier[6]),  # Linear3
    ]
    out_dir = 'output/alexnet'
elif model_type.lower() == 'resnet18':
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()
    layer_names = [
        ('conv1', model.conv1),
        ('layer1', model.layer1[0].conv1),
        ('layer2', model.layer2[0].conv1),
        ('layer3', model.layer3[0].conv1),
        ('layer4', model.layer4[0].conv1),
    ]
    out_dir = 'output/resnet18'
else:
    raise ValueError(f"Unknown model_type: {model_type}. Choose 'alexnet' or 'resnet18'.")

# Move model and input_tensor to the selected device
model = model.to(device)
input_tensor = input_tensor.to(device)

activations = {}

# 3. Register hooks to capture activations
handles = register_hooks(model, layer_names, activations)
# 4. Run a forward pass to collect activations
run_forward(model, input_tensor)

# Load original image as numpy array for overlay (not normalized)
def load_image_for_overlay(img_path, img_size):
    img = Image.open(img_path).convert('RGB').resize(img_size)
    return np.array(img)

orig_image = load_image_for_overlay(img_path, img_size) if overlay_on_image else None

# 5. Visualize and save the activations as images (and optionally as video)
out_dir_used = visualize_activations(
    activations,
    layer_names,
    out_dir,
    show_legend=show_legend,
    save_video=save_video,
    video_filename=video_filename,
    fps=fps,
    cmap=cmap,
    img_size=img_size,
    normalization_mode=normalization_mode,
    use_hist_eq=use_hist_eq,
    clip_percentiles=clip_percentiles,
    save_16bit=save_16bit,
    overlay_on_image=overlay_on_image,
    show_raw_stats=show_raw_stats,
    orig_image=orig_image
)

# 5b. Visualize moving filter if enabled
if show_moving_filter:
    from visualization_utils import visualize_moving_filter
    # For each layer and filter, call visualize_moving_filter
    for layer_name, layer_module in layer_names:
        fmap = activations[layer_name][0]  # (out_channels, h, w)
        num_filters = fmap.shape[0]
        # For first layer, use original image; for others, use previous activation
        if layer_names.index((layer_name, layer_module)) == 0:
            moving_input = orig_image if orig_image is not None else load_image_for_overlay(img_path, img_size)
        else:
            prev_layer_name, _ = layer_names[layer_names.index((layer_name, layer_module)) - 1]
            # Use the mean across all filters of the previous layer as input for visualization
            moving_input = np.mean(activations[prev_layer_name][0].cpu().numpy(), axis=0)
        for filter_idx in range(num_filters):
            visualize_moving_filter(
                layer_module,
                filter_idx,
                moving_input,
                out_dir_used,
                layer_name=layer_name,
                fps=fps,
                cmap=cmap,
                show_legend=show_legend
            )

# 6. Clean up hooks
cleanup_hooks(handles) 