# Semantic Segmentation for Object Identification from Satellite Imagery

## Overview

This project presents an end-to-end pipeline for semantic segmentation of high-resolution satellite imagery, focusing on the Dhaka district of Bangladesh. The primary goal was to develop a deep learning model capable of accurately identifying and classifying objects at the pixel level.

A key contribution of this work is the creation of a novel, meticulously annotated dataset of over 2900 image patches. The project utilizes the DeepLabv3+ architecture and demonstrates excellent performance, offering a robust workflow from raw data collection to model evaluation.

## Key Features

- **Custom Dataset**: A unique, pixel-level annotated dataset of 2900+ 512x512 satellite image patches.
- **High-Performance Model**: Achieved a mIoU of 0.9106 for binary-class segmentation.
- **End-to-End Pipeline**: A complete, 9-step data processing pipeline from image acquisition to model-ready dataset.
- **Reproducibility**: Comprehensive experiment tracking using Weights & Biases across 6+ major training runs.

## The Dhaka Satellite Imagery Dataset

The dataset developed in this project is a core component of the research. It consists of:

- **2900+ Image Patches**: Each resized to 512x512 pixels.
- **High-Resolution Source**: Derived from geotagged `.tiff` satellite imagery.
- **Pixel-Level Annotations**: Each image is paired with a manually created segmentation mask, ideal for supervised learning tasks.
- **Geographical Focus**: All imagery is sourced from the Dhaka district, offering a specialized dataset for urban object analysis.

## Methodology & Pipeline

The project follows a structured, 9-step pipeline:

1. **Data Acquisition**  
   Collecting high-resolution satellite imagery from publicly available sources.

2. **Geospatial Preprocessing**  
   Merging and handling `.tiff` files using QGIS for proper geospatial alignment.

3. **Image Patching**  
   Slicing large satellite maps into 2900+ smaller 512x512 patches.

4. **Data Annotation**  
   Performing meticulous pixel-level labeling for binary classification.

5. **Automated Workflow**  
   Automating dataset splits (train/val/test) using Bash scripts.

6. **Model Selection**  
   Implementing DeepLabv3+ for its superior performance in semantic segmentation tasks.

7. **Accelerated Training**  
   Training conducted on a high-performance server with  
   2 x NVIDIA L40S (48GB) GPUs. Each run lasted over 7 hours.

8. **Experiment Tracking**  
   Using Weights & Biases (wandb) to log metrics, track experiments, and compare 6+ hyperparameter combinations.

9. **Performance Benchmarking**  
   Model evaluated using Mean Intersection over Union (mIoU) and F1-score metrics.
