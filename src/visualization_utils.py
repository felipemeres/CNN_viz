import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import imageio
import torchvision.transforms as T

def load_image(img_path, img_size=(224, 224)):
    """
    Loads an image from disk, resizes it, converts it to a tensor, and normalizes it
    using ImageNet statistics. Returns a tensor suitable for model input.
    """
    img = Image.open(img_path).convert('RGB').resize(img_size)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)
    return input_tensor

def register_hooks(model, layer_names, activations):
    """
    Registers forward hooks on specified layers to capture their outputs (activations)
    during a forward pass. Stores activations in a dictionary keyed by layer name.
    Returns a list of hook handles for later cleanup.
    """
    handles = []
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    for lname, layer in layer_names:
        handles.append(layer.register_forward_hook(get_activation(lname)))
    return handles

def run_forward(model, input_tensor):
    """
    Runs a forward pass through the model with the input tensor, triggering the hooks
    to collect activations.
    """
    with torch.no_grad():
        _ = model(input_tensor)

def cleanup_hooks(handles):
    """
    Removes all registered hooks to avoid memory leaks.
    """
    for h in handles:
        h.remove()

def visualize_activations(
    activations,
    layer_names,
    out_dir,
    show_legend=False,
    save_video=False,
    video_filename=None,
    fps=5,
    cmap='viridis',
    return_frame_paths=False,
    img_size=(224, 224)
):
    """
    For each selected layer and each filter in that layer, normalizes the activation map
    and saves it as a PNG image. If show_legend is True, includes a title with layer/filter info;
    otherwise, saves a clean image. All images are saved in the specified output directory.
    If save_video is True, also saves an uncompressed video of the sequence.
    fps: frames per second for the output video.
    cmap: the matplotlib colormap to use for visualization (default: 'viridis').
    return_frame_paths: if True, also return the list of frame file paths.
    img_size: tuple, the (width, height) in pixels for the output frames and video.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Add date and time to the output directory
    out_dir = os.path.join(out_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    frame_counter = 0
    frame_paths = []  # Collect frame file paths
    width, height = img_size
    dpi = 100
    figsize = (width / dpi, height / dpi)
    for layer, _ in layer_names:
        fmap = activations[layer][0]  # (out_channels, h, w)
        num_filters = fmap.shape[0]
        for idx in range(num_filters):
            fmap_slice = fmap[idx].cpu().numpy()
            # Normalize activation map to [0, 1] for visualization
            fmap_norm = (fmap_slice - np.min(fmap_slice)) / (np.max(fmap_slice) - np.min(fmap_slice) + 1e-5)
            frame_path = os.path.join(out_dir, f"frame_{frame_counter:04d}.png")
            plt.figure(figsize=figsize, dpi=dpi)
            plt.imshow(fmap_norm, cmap=cmap)
            if show_legend:
                plt.title(f'Layer: {layer}, Filter: {idx}')
            plt.axis('off')
            if show_legend:
                plt.tight_layout()
                plt.savefig(frame_path)
            else:
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                plt.savefig(frame_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            frame_paths.append(frame_path)
            frame_counter += 1
    print(f"Frames saved as PNGs in '{out_dir}'.")

    # Save video if requested
    if save_video:
        # If video_filename is not provided, use a default name in the out_dir
        if video_filename is None:
            video_filename = os.path.join(out_dir, "activations.avi")
        else:
            # If a filename (not a path) is provided, save it in out_dir
            if not os.path.isabs(video_filename):
                video_filename = os.path.join(out_dir, video_filename)
        print(f"Saving video to {video_filename} ...")
        frames = [imageio.imread(fp) for fp in frame_paths]
        imageio.mimsave(video_filename, frames, fps=fps, codec='ffv1')
        print(f"Video saved as '{video_filename}'.")
    if return_frame_paths:
        return out_dir, frame_paths
    return out_dir 