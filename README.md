# CNN Activation Visualization

This project provides a unified tool to visualize the activations (feature maps) of convolutional layers in popular neural network models (AlexNet and ResNet18) using PyTorch. The tool saves the activations as images and can optionally create a video of the activations for each filter in the selected layers.

## Features
- Visualize activations from either AlexNet or ResNet18 (pretrained on ImageNet)
- Save each filter's activation as a PNG image (8-bit colormapped or raw 16-bit)
- Optionally save the sequence as a video (lossless AVI)
- Choose colormap, image size, and other visualization options
- **Advanced normalization options:**
  - Per-filter, per-layer, or global normalization
  - Outlier clipping by percentile
  - Histogram equalization for enhanced contrast
- **High-fidelity output:**
  - Save raw 16-bit PNGs for quantitative analysis
- **Overlay and statistics:**
  - Optionally overlay activation heatmaps on the original input image
  - Optionally print and save raw activation statistics (min, max, mean, std)
- Output is organized by model and timestamp for easy comparison

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- numpy
- pillow
- imageio
- scikit-image

Install requirements with:
```bash
pip install torch torchvision matplotlib numpy pillow imageio scikit-image
```

## How to Use

1. **Edit Parameters:**
   Open `src/visualize_model.py` and set the variables at the top of the file:
   - `model_type`: `'alexnet'` or `'resnet18'`
   - `img_path`: Path to your input image
   - `img_size`: Tuple for resizing the input image (e.g., `(600, 600)`)
   - `show_legend`: `True` to include layer/filter info on images, `False` for clean images
   - `save_video`: `True` to save a video, `False` for images only
   - `video_filename`: Name of the output video file (if saving video)
   - `fps`: Frames per second for the video
   - `cmap`: Colormap for visualization (e.g., `'viridis'`, `'plasma'`, `'gray'`)
   - **Advanced options:**
     - `normalization_mode`: `'filter'` (default), `'layer'`, or `'global'`. Controls how normalization is applied:
       - `'filter'`: Each filter is normalized independently (max contrast per filter).
       - `'layer'`: All filters in a layer are normalized together (preserves relative intensity between filters in a layer).
       - `'global'`: All filters in all layers are normalized together (preserves global intensity relationships).
     - `clip_percentiles`: Tuple (low, high), e.g., `(1, 99)`. Clips activation values to this percentile range before normalization to reduce the effect of outliers.
     - `use_hist_eq`: `True` to apply histogram equalization to the normalized activation map for enhanced visual contrast.
     - `save_16bit`: `True` to save raw 16-bit PNGs (for quantitative analysis), `False` to save 8-bit colormapped PNGs (for visualization). Only one type is saved per run.
     - `overlay_on_image`: `True` to overlay the activation heatmap on the original input image (for each filter), `False` for standard visualization.
     - `show_raw_stats`: `True` to print and save min, max, mean, and std statistics for each activation map (saved as `activation_stats.txt` in the output directory).

2. **Run the Script:**
   ```bash
   python src/visualize_model.py
   ```

3. **Find Your Results:**
   - Output images and video will be saved in a timestamped subdirectory under:
     - `output/alexnet/` (for AlexNet)
     - `output/resnet18/` (for ResNet18)
   - Example: `output/alexnet/20240610_153000/frame_0000.png`, `output/alexnet/20240610_153000/activations.avi`
   - If `save_16bit=True`, images will be named `frame_0000_16bit.png`.

## Advanced Normalization and Output Options

### `normalization_mode`
- `'filter'` (default): Each filter's activation map is normalized independently to [0, 1].
- `'layer'`: All filters in a layer are normalized together, preserving their relative strengths.
- `'global'`: All filters in all layers are normalized together, preserving global intensity relationships.

### `clip_percentiles`
- Clips activation values to the specified percentile range (e.g., `(1, 99)`) before normalization. This reduces the influence of extreme outliers and can improve visualization.

### `use_hist_eq`
- If `True`, applies histogram equalization to the normalized activation map, enhancing contrast and making subtle features more visible.

### `save_16bit`
- If `True`, saves each activation map as a raw 16-bit PNG (grayscale, no colormap), suitable for quantitative analysis or further processing. The image is resized to `img_size`.
- If `False`, saves each activation map as an 8-bit PNG with the selected colormap (for human interpretation).
- **Note:** Only one type of image is saved per run.

### `overlay_on_image`
- If `True`, overlays the activation heatmap (colormap) on top of the original input image for each filter, blending the two for easier interpretation of which regions are most active.
- If `False`, shows only the activation map.
- The original image is resized to match the activation map if needed.

### `show_raw_stats`
- If `True`, prints and saves the minimum, maximum, mean, and standard deviation of the raw activation values for each filter to `activation_stats.txt` in the output directory.
- Useful for quantitative analysis and debugging.
- If `False`, no statistics are printed or saved.

## How the Colors Work
- The colormap (e.g., `'viridis'`, `'plasma'`) maps normalized activation values to colors for human interpretation.
- The colors are not part of the model's learning process; they are for visualization only.
- You can change the colormap by setting the `cmap` variable in `visualize_model.py`.

## Customization
- To add more models or layers, simply extend the logic in `visualize_model.py` and/or `visualization_utils.py`.
- The code is modular and easy to adapt for other PyTorch models.

## License
MIT License

## How Are the Colors in the Activation Visualizations Determined?

The colors of the pixels in the activation images saved by the visualization scripts are determined by the values of the feature maps (activations) produced by the neural network's filters when processing your input image.

### How are the colors determined?
- Each filter in a convolutional layer produces a 2D array (feature map) of activation values for a given input.
- In the scripts, these values are **normalized** to the range [0, 1] for visualization (see normalization options above).
- The `matplotlib` function `imshow(..., cmap='viridis')` is used to display these values as an image, mapping low values to one color (e.g., dark blue) and high values to another (e.g., yellow), according to the chosen colormap (`'viridis'`).

### Are the colors part of the learning process?
- **No, the colors themselves are not part of the learning process.**
- The **activation values** (the numbers in the feature maps) are the result of the learning process: they depend on the learned weights of the filters and the input image.
- The **color mapping** is just a way for humans to visualize these numbers. You could use a different colormap (e.g., `'gray'`, `'plasma'`, etc.) and the underlying data would be the same.

### Are the colors random?
- **No, they are not random.**
- The color of each pixel in the visualization is directly determined by the normalized activation value at that location in the feature map, mapped through the chosen colormap.

---

**Summary:**
- The colors are a visualization of the learned filter responses to your input image.
- The mapping from activation value to color is arbitrary and for human interpretation only.
- The actual values (and thus the patterns you see) are a direct result of the network's learning, but the colors themselves are not learned or random—they are chosen by the visualization code. 