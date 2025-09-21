# Enhanced Parameter Study for YOLO Post-processing

This repository contains the supplementary code for the paper: *"Your Paper Title Here"*. It implements an efficient parameter study designed to optimize a sophisticated YOLO post-processing pipeline which incorporates tiling, spatial gating, semantic gating, and confidence boosting.

## Overview

The parameter study employs a reduced-cost approach to efficiently navigate the large configuration space:

1.  **Two-Stage Search:** A coarse grid search (Stage A) followed by a focused random search (Stage B), both executed on small image subsets.
2.  **Finalization:** Re-evaluation of the top-K configurations on a larger evaluation set.
3.  **Ablation Study:** Systematic analysis of the contribution of each pipeline component.

## Features

*   **Efficient Search Strategy:** Minimizes computational cost.
*   **Caching Mechanism:** Encapsulated caching avoids redundant YOLO inferences.
*   **Publication-Ready Visualization:** Generates IEEE-style PDF plots and tables.
*   **Reproducible Setup:** Clear dependency management and command-line interface.

## Setup

### Prerequisites

*   Python 3.8+
*   NVIDIA GPU (highly recommended) and CUDA drivers.

### Installation

1.  Clone the repository:
    ```bash
    git clone <your-repository-url>
    cd parameter-study-yolo
    ```

2.  Create a virtual environment (recommended) and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

### Data Preparation

The script expects the VisDrone dataset format (or similar).

1.  **Dataset:** Place evaluation images and annotations in separate directories.
2.  **Model Weights:** Obtain a trained YOLOv8 model weight file (e.g., `best.pt`).

## Usage

The parameter study is executed via the command line using `run_study.py`.

```bash
python run_study.py --weights path/to/your/weights.pt \
                    --img_dir path/to/your/images \
                    --annotation_dir path/to/your/annotations \
                    [--output_dir ./results] \
                    [--subset_a 15] \
                    [--full_eval_size 100]