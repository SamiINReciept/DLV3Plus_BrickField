# Semantic Segmentation for Object Identification from Satellite Imagery (DLV3Plus_BrickField)

📖 Overview
This project presents an end-to-end pipeline for semantic segmentation of high-resolution satellite imagery, with a focus on the Dhaka district of Bangladesh. The primary goal was to develop a deep learning model capable of accurately identifying and classifying objects at the pixel level.

A key contribution of this work is the creation of a novel, meticulously annotated dataset of 2900+ image patches. The project utilizes the DeepLabv3+ architecture and achieves excellent performance, demonstrating a robust workflow from raw data collection to model evaluation.

✨ Key Features
Custom Dataset: A unique, pixel-level annotated dataset of 2900+ 512x512 satellite image patches.

High-Performance Model: Achieved a mIoU of 0.9106 for binary-class segmentation.

End-to-End Pipeline: A complete, 9-step data processing pipeline from image acquisition to a model-ready dataset.

Reproducibility: Comprehensive experiment tracking using Weights & Biases for over 6 major training runs.

🏞️ The Dhaka Satellite Imagery Dataset
The dataset developed in this project is a core component of this research. It consists of:

2900+ Image Patches: Sliced into 512x512 pixel resolution.

High-Resolution Source: Derived from geotagged (.tiff) satellite imagery.

Pixel-Level Annotations: Each image is paired with a manually created segmentation mask, making it suitable for supervised deep learning tasks.

Geographical Focus: All images are from the Dhaka district, providing a specialized dataset for urban object analysis.

⚙️ Methodology & Pipeline
The project was executed through a systematic 9-step pipeline:

Data Acquisition: Collecting high-resolution satellite imagery.

Geospatial Preprocessing: Merging and handling geotagged .tiff files using qGIS.

Image Patching: Slicing large satellite maps into 2900+ smaller 512x512 patches.

Data Annotation: Performing meticulous pixel-level labeling for binary classification.

Automated Workflow: Using Bash scripts to automate the creation of training, validation, and test sets.

Model Selection: Implementing the DeepLabv3+ architecture for its effectiveness in semantic segmentation.

Accelerated Training: Leveraging a high-performance server with 2 x NVIDIA L40S (48GB) GPUs. Each training run lasted over 7 hours.

Experiment Tracking: Using Weights & Biases (Wandb) to log metrics, compare 6+ hyperparameter combinations, and visualize results.

Performance Benchmarking: Evaluating the model using Mean Intersection over Union (mIoU) and F1-score metrics.
