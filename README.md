# YOLO11-seg Active Learning-based Annotation Tool

This is a comprehensive GUI-based image annotation tool developed with Python and PyQt5, specifically optimized for **YOLOv11-seg** models. It integrates an **Active Learning** workflow to significantly accelerate object segmentation tasks by combining manual annotation tools with AI-assisted labeling and refinement.

## ✨ Key Features

### 1. 🧠 Active Learning & Automation
- **Auto-Labeling (Inference):** Automatically detect and segment objects in your images using a pre-trained YOLOv11-seg model.
- **Class Specification (Refinement):** A specialized feature to refine specific labels. Select a "target class" (e.g., generic "vehicle"), and use a secondary, specialized model to classify it into more specific categories (e.g., "car", "bus", "truck") by analyzing cropped regions.

### 2. 🖋️ Versatile Annotation Tools
- **Polygon Tool (W):** precise manual annotation. Click to add points, right-click to close. Supports editing (move vertices, delete vertices).
- **Paint Brush Tool (B):** Draw masks naturally like a painting app. 
    - **Smart Simplification (RDP):** Uses the RDP algorithm to automatically simplify your brush strokes into clean polygons. You can adjust the smoothness (`Epsilon`) in real-time.
    - **Eraser Mode:** Switch to eraser (`E`) to remove parts of a shape or split it.
- **SAM (Segment Anything Model) Integration (S):** Interactively segment objects by simply clicking positive (Left click) and negative (Right click or `E`) points.

### 3. 🏷️ Advanced Class Management
- **Class Manager:** A dedicated dialog to **Add**, **Rename**, and **Delete** classes. 
    - **Smart Renaming:** Renaming a class automatically updates all existing labels in the current session.
    - **Safe Deletion:** When deleting a class, you can choose to remove all associated polygon instances from the current image.
- **Class Selector:** Quick popup to assign classes when creating new shapes.
- **Color Coding:** Automatically generates distinct colors for each class.

### 4. 🚀 Model Training & Dataset Management
- **Integrated Training:** Fine-tune your YOLOv11 model directly within the app.
    - **Hyperparameters:** Configure epochs, batch size, learning rate, optimizer, etc.
    - **Augmentations:** Set geometry (flip, scale, shear) and color (HSV) augmentations.
- **YAML Creator:** Built-in tool to create dataset `data.yaml` files. Simply select your train/val image folders and input class names.
- **Auto-Update:** After training, the application automatically reloads the new "best" model for immediate feedback.

### 5. 📂 Data Management
- **Flexible Import:** Open any folder of images. Labels are automatically detected if they exist in a sibling `labels` directory (e.g., `images/..`, `labels/..`).
- **Upload Labels:** Manually import labels from a specific source directory if your structure differs.
- **Export:** Export your finalized dataset (images + YOLO format txt labels) to a clean directory structure, ready for MLOps pipelines.

## ⚙️ Requirements

- Python 3.8+
- PyQt5
- ultralytics (YOLOv11)
- numpy
- rdp
- opencv-python-headless
- torch
- torchvision
- shapely
- PyYAML
- onnxruntime (for SAM)

## 🚀 Installation

We recommend using [uv](https://github.com/astral-sh/uv) for fast and reliable package management.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/hyeok90/PyQt_active_learning
    cd PyQt_active_learning
    ```

2.  **Install uv (if not already installed):**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3.  **Create and Activate Virtual Environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```

4.  **Install PyTorch (with CUDA support):**
    To ensure you get the GPU-enabled version of PyTorch (e.g., CUDA 12.1), run:
    ```bash
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    ```
    *If you only need CPU support, you can skip this step.*

5.  **Install other dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

6.  **Fix OpenCV Compatibility:**
    ```bash
    uv pip uninstall opencv-python
    uv pip install --force-reinstall opencv-python-headless
    ```

## 🏃‍♂️ Usage Workflow

1.  **Start the App:**
    ```bash
    python main.py
    ```

2.  **Load Data:**
    - **Open Image Folder:** `File -> Open Image Folder`. Labels in `../labels` are loaded automatically.
    - **(Optional) Upload Labels:** If you have existing labels elsewhere, use `File -> Upload Labels`.

3.  **Active Learning (Auto-Labeling):**
    - **Load Model:** `File -> Run Model Inference`. Select a `.pt` model and images.
    - **Refine:** Use **Class Specification** (`Edit -> Class Specification`) to refine labels using a second model if needed.

4.  **Manual Annotation:**
    - **Draw:** Press `W` for Polygon or `B` for Paint.
    - **SAM:** Press `S` to enter SAM mode. Click to segment. Press `F` to accept.
    - **Edit:** Click a polygon to select. Drag points to move. Press `Delete` to remove.

5.  **Train:**
    - **Create Config:** `File -> Train`. Click "Create New..." to generate a YAML file from your current data folders.
    - **Run Training:** Set parameters and start. Watch the console for progress. The model updates automatically upon completion.

6.  **Export:**
    - `File -> Export`. Save your clean dataset for deployment or further training.

## ⌨️ Shortcuts

### Global / Navigation
| Key | Action |
| :--- | :--- |
| `A` / `D` | Previous / Next Image |
| `Space` | Fit Image to Window |
| `Ctrl+S` | Save Current Labels |
| `Ctrl+Z` | Undo Last Shape Modification |
| `Delete` | Delete Selected Instance(s) |
| `Mouse Wheel` | Zoom In / Out |
| `Middle Drag` | Pan Image |

### Tools (Modes)
| Key | Action |
| :--- | :--- |
| `W` | Toggle **Polygon** Mode |
| `B` | Toggle **Paint** Mode |
| `S` | Toggle **SAM** Mode |

### Paint Mode (B)
| Key | Action |
| :--- | :--- |
| `Q` | Switch to **Brush** |
| `E` | Switch to **Eraser** |
| `[` / `]` | Decrease / Increase Brush Size |
| `,` / `.` | Decrease / Increase Smoothness (Epsilon) |

### SAM Mode (S)
| Key | Action |
| :--- | :--- |
| `Q` | Add Positive Point (Green) |
| `E` | Add Negative Point (Red) |
| `G` | Clear All Points |
| `F` | Finalize & Create Shape |