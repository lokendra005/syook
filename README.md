# PPE Detection System

## Overview

This project implements an advanced Personal Protective Equipment (PPE) detection system using YOLOv8. The system is designed to enhance safety monitoring in construction sites and similar environments by detecting persons and various types of PPE in images.

## Objectives

- Develop two YOLOv8 models:
  1. Person detection model
  2. PPE detection model
- Process images through both models sequentially
- Detect and visualize PPE usage on persons in images

## Features

- Detects persons in images
- Identifies PPE items: hard-hats, gloves, masks, glasses, boots, vests, PPE suits, ear-protectors, and safety harnesses
- Converts annotations from Pascal VOC to YOLO format
- Separates person and PPE labels for individual model training
- Crops person images for PPE detection
- Performs inference using both person and PPE detection models

## Installation

1. Clone this repository:
2. Install the required dependencies:
   
## Usage

### 1. Annotation Conversion

Convert Pascal VOC annotations to YOLO format:
### 2. Label Separation

Separate person and PPE labels:
### 3. Dataset Preparation

Prepare the dataset for PPE detection:
### 4. Inference

Run inference on images:


## Detailed documentation 
For a detailed explanation of the project refer to this [Google Document](https://docs.google.com/document/d/1B_Xdm-JCnRapn2-wut4AHbkOV1GgZSkldi5_tqkEhC0/edit#heading=h.7bwgoug0lr4).
