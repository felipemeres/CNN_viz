# CNN Activation Visualization

This project provides a unified tool to visualize the activations (feature maps) of convolutional layers in popular neural network models (AlexNet and ResNet18) using PyTorch. The tool saves the activations as images and can optionally create a video of the activations for each filter in the selected layers.

## Features
- Visualize activations from either AlexNet or ResNet18 (pretrained on ImageNet)
- Save each filter's activation as a PNG image
- Optionally save the sequence as a video (lossless AVI)
- Choose colormap, image size, and other visualization options
- Output is organized by model and timestamp for easy comparison

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- numpy
- pillow
- imageio

Install requirements with:
```bash
pip install torch torchvision matplotlib numpy pillow imageio
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

2. **Run the Script:**
   ```bash
   python src/visualize_model.py
   ```

3. **Find Your Results:**
   - Output images and video will be saved in a timestamped subdirectory under:
     - `output/alexnet/` (for AlexNet)
     - `output/resnet18/` (for ResNet18)
   - Example: `output/alexnet/20240610_153000/frame_0000.png`, `output/alexnet/20240610_153000/activations.avi`

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
- In the scripts, these values are **normalized** to the range [0, 1] for visualization.
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