import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from visualization_utils import load_image, register_hooks, run_forward, cleanup_hooks, visualize_activations

# --------------------
# User Parameters
# --------------------
model_type = 'resnet18'  # 'alexnet' or 'resnet18'
img_path = 'V:/AI/datasets/imagenet/imagenet21k_resized.tar/imagenet21k_resized/imagenet21k_train/n02104280_housedog/n02104280_853.JPEG'
img_size = (600, 600)  # Image resize dimensions
show_legend = False  # True to include text/legend, False for clean images
save_video = True  # True to save video
video_filename = 'activations.avi'  # Output video file (if saving video)
fps = 24  # Frames per second for the video
cmap = 'plasma'  # Colormap for visualization (e.g., 'viridis', 'gray', 'plasma', etc.)
# --------------------

# 1. Load and preprocess the image
input_tensor = load_image(img_path, img_size)

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
    cmap=cmap
)
# 6. Clean up hooks
cleanup_hooks(handles) 