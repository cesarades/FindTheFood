# Find The Food

## Overview

This project is an empirical analysis of zero-shot object detection models for food segmentation, benchmarked on the FoodSeg103 dataset.

## Table of Contents

1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Methodology](#methodology)
4. [Models Evaluated](#models-evaluated)
5. [Metrics](#metrics)
6. [Requirements](#requirements)
7. [Installation](#installation)
8. [Dataset](#dataset)
9. [Usage](#usage)
    - [Inference](#inference)
    - [Metrics Calculation](#metrics-calculation)
    - [Data Visualization](#data-visualization)
10. [Results](#results)
11. [Contributing](#contributing)
12. [License](#license)
13. [References](#references)
14. [Contact](#contact)

## Introduction

Semantic segmentation of food items in images is a challenging task due to the high variability and overlapping nature of food items. This project explores the use of zero-shot object detection models combined with advanced segmentation techniques to perform food segmentation without any additional model training.

## Key Features

- **Zero-Shot Object Detection**: Utilizes pre-trained models without fine-tuning.
- **Mask Combination Method**: Combines segmentation masks based on the most frequent food per pixel, resolving ties with the highest confidence detection.
- **Polygon Refinement**: Enhances segmentation masks with polygon refinement for better accuracy.
- **Benchmarking**: Evaluated on the FoodSeg103 dataset.

## Methodology

The proposed method involves the following steps:

1. **Object Detection**: Use an out-of-the-box zero-shot object detection model to detect food items in images.
2. **Segmentation**: Pass the resulting bounding boxes to the Segment Anything Model (SAM) to obtain segmentation masks.
3. **Polygon Refinement**: Refine the segmentation masks using polygon-based methods.
4. **Mask Combination**: Combine all segmentation masks using the proposed method.


## Models Evaluated

The object detection models tested in this study include:

- **OWL-ViT**: Base and Large versions.
- **OWLv2**: Base and Large versions.
- **OmDet**

Initial tests were conducted with Grounding DINO (base and tiny), but this model class was not well suited for this project specific framework.

## Metrics

The models were assessed based on the following metrics:

- **mIoU**: Mean Intersection over Union over each class.
- **mACC**: Mean accuracy over all classes.
- **aAcc**: Overall pixel accuracy.

## Requirements

- **Operating System**: Linux
- **GPU**: CUDA-compatible GPU
- **Python Version**: Python 3.7 or higher
- **Package Manager**: Conda

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/find-the-food.git
   cd find-the-food
   ```

2. **Create and Activate Conda Environment**

   ```bash
   conda env create -f environment.yml
   conda activate find-the-food
   ```

   This will install all required packages as specified in the `environment.yml` file.

## Dataset

We tested our framework on the [FoodSeg103 dataset](https://xiongweiwu.github.io/foodseg103.html). The test set is already included in this repository.

1. **FoodSeg103**
   ```
   FindTheFood 
   └── data
       └── (your dataset)
       └── FoodSeg103
           └── Images
               ├── ann_dir   # Ground truth masks
               │   ├── 00000001.png
               │   └── ...
               ├── img_dir   # RGB images
               │   ├── 00000001.jpg
               │   └── ...
               └── inf_dir   # Outputs (will be generated)
   ```

   - **ann_dir**: Contains the ground truth masks.
   - **img_dir**: Contains the RGB images.
   - **inf_dir**: Will contain the outputs from the models.

2. **Custom Dataset**

   To test on another dataset, ensure that it follows the same directory structure as above. 

## Usage

### Inference

Run the inference script to generate segmentation masks:

```bash
python inference.py --model <model_index> --threshold <value> --nms <value> --polygon_refin <True/False> --topk <value> --dataset <dataset_name>
```

**Model Index:**

0. google/owlv2-base-patch16-ensemble
1. google/owlv2-large-patch14-ensemble
2. google/owlvit-base-patch32
3. google/owlvit-large-patch14
4. omlab/omdet-turbo-swin-tiny-hf
5. IDEA-Research/grounding-dino-tiny
6. IDEA-Research/grounding-dino-base

**Default Values:**

- `--model`: 0
- `--threshold`: 0.0
- `--nms`: 0.3
- `--polygon_refin`: True
- `--topk`: None
- `--dataset`: FoodSeg103

**Parameters:**

- `--model`: Model index (0-6 inclusive).
- `--threshold`: Confidence threshold for detections.
- `--nms`: Non-Maximum Suppression threshold.
- `--polygon_refin`: Whether to use polygon refinement (True/False).
- `--topk`: Number of top detections to consider.
- `--dataset`: Name of the dataset.

**Example:**

To run inference using model 2 with default parameters on the FoodSeg103 dataset:

```bash
python inference.py --model 2 --dataset FoodSeg103
```

### Metrics Calculation

Metrics are automatically calculated after inference and stored in `stored_metrics.csv`. If you would like to recalculate metrics for all or a subset of models run:


```bash
python data_analysis.py --model <model_index> --threshold <value> --nms <value> --polygon_refin <True/False> --topk <value> --dataset <dataset_name>
```

**Example:**

To calculate metrics on all the infered models, run:

```bash
python data_analysis.py 
```

### Data Visualization

Visualize the data and results:

```bash
python data_visualization.py
```

## Results

[Include your results here, such as tables, charts, or example images demonstrating the performance of your method.]


## References
- [Hugging Face](https://huggingface.co/models?pipeline_tag=zero-shot-object-detection&sort=downloads)
- [Segment Anything Model (SAM)](https://arxiv.org/abs/2308.05938)
- [OWL-ViT](https://arxiv.org/abs/2205.06230)
- [OWLv2](https://arxiv.org/abs/2306.09683)
- [OmDet](https://arxiv.org/abs/2403.06892)
- [FoodSeg103 Dataset](https://arxiv.org/abs/2105.05409)
