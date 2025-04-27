import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import imageio
import torchvision.transforms as T
from skimage import exposure  # Add this import for histogram equalization
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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
    show_raw_stats=False,    # Show/save raw stats for each activation map
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
                overlay_alpha = 0.9
                overlay = (orig_image / 255.0) * (1 - overlay_alpha) + heatmap * overlay_alpha
                overlay = np.clip(overlay, 0, 1)
                plt.imshow(overlay)
            else:
                if np.isscalar(fmap_norm) or getattr(fmap_norm, 'ndim', 1) == 0:
                    plt.imshow(np.array([[fmap_norm]]), cmap=cmap)
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
        # Use macro_block_size=1 to avoid resizing warnings/errors
        imageio.mimsave(video_filename, frames, fps=fps, codec='ffv1', macro_block_size=1)
        print(f"Video saved as '{video_filename}'.")
    if return_frame_paths:
        return out_dir, frame_paths
    return out_dir

def visualize_moving_filter(layer_module, filter_idx, moving_input, out_dir, layer_name=None, fps=5, cmap='viridis', show_legend=True):
    """
    Visualize the moving filter for a given layer/filter and input (image or activation map).
    Saves a video in the output directory.
    If show_legend is False, subplot titles and activation values are omitted.
    """
    # Get filter weights (for Conv2d)
    if hasattr(layer_module, 'weight'):
        # For first layer, weights shape: (out_channels, in_channels, kH, kW)
        # For later layers, in_channels may > 1
        weights = layer_module.weight.data.cpu().numpy()[filter_idx]
    else:
        print(f"Layer {layer_name} has no weights to visualize.")
        return

    # Prepare input: if 3D (C, H, W), take mean across channels for visualization
    if moving_input.ndim == 3:
        if moving_input.shape[0] == 3 or moving_input.shape[0] == 1:  # (C, H, W)
            input_vis = np.mean(moving_input, axis=0)
        else:  # (H, W, C)
            input_vis = np.mean(moving_input, axis=-1)
    else:
        input_vis = moving_input
    input_vis = (input_vis - np.min(input_vis)) / (np.max(input_vis) - np.min(input_vis) + 1e-8)
    input_vis = (input_vis * 255).astype(np.uint8)
    h, w = input_vis.shape

    # Get filter size
    if weights.ndim == 3:
        kH, kW = weights.shape[1:]
    elif weights.ndim == 2:
        kH, kW = weights.shape
    else:
        print(f"Unexpected filter shape: {weights.shape}")
        return

    # Prepare activation map (output of convolution)
    stride = layer_module.stride[0] if hasattr(layer_module, 'stride') else 1
    pad = layer_module.padding[0] if hasattr(layer_module, 'padding') else 0
    out_h = (h + 2 * pad - kH) // stride + 1
    out_w = (w + 2 * pad - kW) // stride + 1
    activation_map = np.zeros((out_h, out_w))

    # Pad input for visualization if needed
    input_padded = np.pad(input_vis, pad, mode='constant') if pad > 0 else input_vis

    # Prepare colormap
    cmap_func = plt.get_cmap(cmap)

    frames = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    font_thickness = 1
    bg_color = (0, 0, 0)
    pad_px = 5
    for i in range(out_h):
        for j in range(out_w):
            window = input_padded[i*stride:i*stride+kH, j*stride:j*stride+kW]
            if weights.ndim == 3:
                act_val = np.sum(window * np.mean(weights, axis=0))
            else:
                act_val = np.sum(window * weights)
            activation_map[i, j] = act_val

            # 1. Input with filter window
            input_disp = cv2.cvtColor(input_vis, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(input_disp, (j*stride, i*stride), (j*stride+kW-1, i*stride+kH-1), (0,0,255), 2)
            if show_legend:
                cv2.putText(input_disp, 'Input with Filter Window', (pad_px, 15), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

            # 2. Magnified filter window
            window_disp = window.copy()
            window_disp = (window_disp - window_disp.min()) / (np.ptp(window_disp) + 1e-8)
            window_disp = (window_disp * 255).astype(np.uint8)
            window_disp = cv2.resize(window_disp, (input_vis.shape[1], input_vis.shape[0]), interpolation=cv2.INTER_NEAREST)
            window_disp = cv2.cvtColor(window_disp, cv2.COLOR_GRAY2BGR)
            if show_legend:
                cv2.putText(window_disp, 'Filter Window (Zoomed)', (pad_px, 15), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

            # 3. Filter weights
            filt = np.mean(weights, axis=0) if weights.ndim == 3 else weights
            filt_disp = (filt - filt.min()) / (np.ptp(filt) + 1e-8)
            filt_disp = (filt_disp * 255).astype(np.uint8)
            filt_disp = cv2.resize(filt_disp, (input_vis.shape[1], input_vis.shape[0]), interpolation=cv2.INTER_NEAREST)
            filt_disp = cv2.applyColorMap(filt_disp, cv2.COLORMAP_BWR if hasattr(cv2, 'COLORMAP_BWR') else cv2.COLORMAP_JET)
            if show_legend:
                cv2.putText(filt_disp, 'Filter Weights', (pad_px, 15), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

            # 4. Activation map so far
            act_map_norm = (activation_map - np.min(activation_map)) / (np.max(activation_map) - np.min(activation_map) + 1e-8)
            act_disp = (act_map_norm * 255).astype(np.uint8)
            act_disp = cv2.resize(act_disp, (input_vis.shape[1], input_vis.shape[0]), interpolation=cv2.INTER_NEAREST)
            act_disp = cv2.applyColorMap(act_disp, cv2.COLORMAP_BONE)
            if show_legend:
                cv2.putText(act_disp, f'Activation Map (step {i*out_w+j+1})', (pad_px, 15), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                cv2.putText(act_disp, f'{act_val:.2f}', (j*stride, i*stride+20), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

            # Concatenate horizontally
            frame = cv2.hconcat([input_disp, window_disp, filt_disp, act_disp])
            frames.append(frame)

    video_filename = f"moving_filter_{layer_name}_filter{filter_idx:03d}.avi"
    video_path = os.path.join(out_dir, video_filename)
    print(f"Saving moving filter video to {video_path} ...")
    imageio.mimsave(video_path, frames, fps=fps, codec='ffv1', macro_block_size=1)
    print(f"Moving filter video saved: {video_path}") 