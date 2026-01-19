# STC Viewer

Real-time 3D visualization of MNE source estimates using WebGPU.

## Features

- **High-Performance Rendering**: Utilizes WebGPU (via `wgpu-py`) for smooth 3D visualization.
- **Dynamic Animation**: Visualize source time courses compatible with MNE data formats.
- **Interactive Controls**: Orbit, pan, and zoom around the brain model.
- **Atlas Visualization**: Toggle between dynamic source activation and atlas region colors.
- **Trace Overlay**: Optional "butterfly plot" overlay for time traces.

## Installation

1. Ensure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   **Dependencies:**
   - `wgpu`
   - `rendercanvas`
   - `glfw`
   - `numpy`
   - `pyrr`
   - `trimesh`
   - `requests`

## Usage

Run the main application script:

```bash
python main.py
```

## Controls

### Mouse
- **Left Click + Drag**: Orbit / Rotate
- **Right Click + Drag**: Pan
- **Scroll**: Zoom

### Keyboard
- **`t`**: Toggle between **Dynamic Source Animation** and **Surface Atlas Mode**.
- **`p`**: Toggle the **Butterfly Plot Overlay**.
