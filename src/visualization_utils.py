import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import imageio
import torchvision.transforms as T
from skimage import exposure  # Add this import for histogram equalization

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
    img_size=(224, 224),
    normalization_mode='filter',  # 'filter', 'layer', or 'global'
    use_hist_eq=False,
    clip_percentiles=None,  # e.g., (1, 99) to clip outliers
    save_16bit=False,  # Save 16-bit PNGs of raw activations
    overlay_on_image=False,  # Overlay heatmap on input image
    show_raw_stats=False,    # Show/save raw stats for each activation
    orig_image=None          # Original image as a numpy array (H, W, 3), unnormalized, resized
):
    """
    For each selected layer and each filter in that layer, normalizes the activation map
    and saves it as a PNG image. If show_legend is True, includes a title with layer/filter info;
    otherwise, saves a clean image. All images are saved in the specified output directory.
    If save_video is True, also saves an uncompressed video of the sequence.
    normalization_mode: 'filter' (default), 'layer', or 'global'.
    fps: frames per second for the output video.
    cmap: the matplotlib colormap to use for visualization (default: 'viridis').
    return_frame_paths: if True, also return the list of frame file paths.
    img_size: tuple, the (width, height) in pixels for the output frames and video.
    use_hist_eq: if True, use histogram equalization for normalization.
    clip_percentiles: tuple (low, high) to clip outliers before normalization.
    save_16bit: if True, save raw 16-bit PNGs of the normalized activations.
    overlay_on_image: if True, overlay activation heatmap on input image.
    show_raw_stats: if True, print/save min, max, mean, std for each activation map.
    orig_image: original input image as numpy array (H, W, 3), for overlay.
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
    stats_lines = []

    # --- Compute normalization stats ---
    # Gather all activations if needed for global/layer normalization
    global_min, global_max = None, None
    layer_mins, layer_maxs = {}, {}
    if normalization_mode == 'global':
        all_vals = []
        for layer, _ in layer_names:
            fmap = activations[layer][0]
            all_vals.append(fmap.cpu().numpy().flatten())
        all_vals = np.concatenate(all_vals)
        if clip_percentiles:
            low, high = np.percentile(all_vals, clip_percentiles)
            all_vals = np.clip(all_vals, low, high)
        global_min, global_max = np.min(all_vals), np.max(all_vals)
    elif normalization_mode == 'layer':
        for layer, _ in layer_names:
            fmap = activations[layer][0]
            vals = fmap.cpu().numpy().flatten()
            if clip_percentiles:
                low, high = np.percentile(vals, clip_percentiles)
                vals = np.clip(vals, low, high)
            layer_mins[layer], layer_maxs[layer] = np.min(vals), np.max(vals)

    for layer, _ in layer_names:
        fmap = activations[layer][0]  # (out_channels, h, w)
        num_filters = fmap.shape[0]
        for idx in range(num_filters):
            fmap_slice = fmap[idx].cpu().numpy()
            # --- Outlier clipping ---
            if clip_percentiles:
                low, high = np.percentile(fmap_slice, clip_percentiles)
                fmap_slice = np.clip(fmap_slice, low, high)
            # --- Normalization ---
            if normalization_mode == 'global' and global_min is not None and global_max is not None:
                norm_min, norm_max = global_min, global_max
            elif normalization_mode == 'layer' and layer in layer_mins:
                norm_min, norm_max = layer_mins[layer], layer_maxs[layer]
            else:  # 'filter' (default)
                norm_min, norm_max = np.min(fmap_slice), np.max(fmap_slice)
            # Avoid division by zero
            denom = (norm_max - norm_min) if (norm_max - norm_min) > 1e-8 else 1e-8
            fmap_norm = (fmap_slice - norm_min) / denom
            # --- Histogram equalization ---
            if use_hist_eq:
                fmap_norm = exposure.equalize_hist(fmap_norm)
            # --- Show/save raw stats ---
            if show_raw_stats:
                stats = {
                    'layer': layer,
                    'filter': idx,
                    'min': float(np.min(fmap_slice)),
                    'max': float(np.max(fmap_slice)),
                    'mean': float(np.mean(fmap_slice)),
                    'std': float(np.std(fmap_slice))
                }
                stats_line = f"Layer: {layer}, Filter: {idx}, min: {stats['min']:.4f}, max: {stats['max']:.4f}, mean: {stats['mean']:.4f}, std: {stats['std']:.4f}"
                print(stats_line)
                stats_lines.append(stats_line)
            # --- Save 16-bit PNG (raw, not colormapped) ---
            if save_16bit:
                fmap_16bit = (fmap_norm * 65535).astype(np.uint16)
                raw16_path = os.path.join(out_dir, f"frame_{frame_counter:04d}_16bit.png")
                Image.fromarray(fmap_16bit).save(raw16_path)
            # --- Save visual (colormapped) PNG ---
            frame_path = os.path.join(out_dir, f"frame_{frame_counter:04d}.png")
            plt.figure(figsize=figsize, dpi=dpi)
            if overlay_on_image and orig_image is not None:
                # Resize fmap_norm to match orig_image
                fmap_resized = np.array(Image.fromarray((fmap_norm * 255).astype(np.uint8)).resize((orig_image.shape[1], orig_image.shape[0]), resample=Image.BILINEAR)) / 255.0
                # Get colormap heatmap (RGBA)
                cmap_func = plt.get_cmap(cmap)
                heatmap = cmap_func(fmap_resized)[:, :, :3]  # Drop alpha
                # Blend heatmap with original image
                overlay_alpha = 0.1
                overlay = (orig_image / 255.0) * (1 - overlay_alpha) + heatmap * overlay_alpha
                overlay = np.clip(overlay, 0, 1)
                plt.imshow(overlay)
            else:
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
    # Save stats to file if requested
    if show_raw_stats and stats_lines:
        stats_path = os.path.join(out_dir, 'activation_stats.txt')
        with open(stats_path, 'w') as f:
            for line in stats_lines:
                f.write(line + '\n')
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