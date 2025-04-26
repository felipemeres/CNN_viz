import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from visualization_utils import load_image, register_hooks, run_forward, cleanup_hooks, visualize_activations
import os
from dotenv import load_dotenv

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
        ('features_0', model.features[0]),  # Conv1
        ('features_3', model.features[3]),  # Conv2
        ('features_6', model.features[6]),  # Conv3
        ('features_8', model.features[8]),  # Conv4
        ('features_10', model.features[10]), # Conv5
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
    save_16bit=save_16bit
)
# 6. Clean up hooks
cleanup_hooks(handles) 