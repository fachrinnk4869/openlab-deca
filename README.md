# DECA: Detailed Expression Capture and Animation (SIGGRAPH2021)

<p align="center"> 
<img src="TestSamples/teaser/results/teaser.gif">
</p>
<p align="center">input image, aligned reconstruction, animation with various poses & expressions<p align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YadiraF/DECA/blob/master/Detailed_Expression_Capture_and_Animation.ipynb?authuser=1)

This is the official Pytorch implementation of DECA. 

DECA reconstructs a 3D head model with detailed facial geometry from a single input image. The resulting 3D head model can be easily animated. Please refer to the [arXiv paper](https://arxiv.org/abs/2012.04012) for more details.

The main features:

* **Reconstruction:** produces head pose, shape, detailed face geometry, and lighting information from a single image.
* **Animation:** animate the face with realistic wrinkle deformations.
* **Robustness:** tested on facial images in unconstrained conditions.  Our method is robust to various poses, illuminations and occlusions. 
* **Accurate:** state-of-the-art 3D face shape reconstruction on the [NoW Challenge](https://ringnet.is.tue.mpg.de/challenge) benchmark dataset.
  
## Getting Started
Clone the repo:
  ```bash
  git clone https://github.com/YadiraF/DECA
  cd DECA
  ```  

### Requirements
* Python 3.7 (numpy, skimage, scipy, opencv)  
* PyTorch >= 1.6 (pytorch3d)  
* face-alignment (Optional for detecting face)  
  You can run 
  ```bash
  pip install -r requirements.txt
  ```
  Or use virtual environment by runing 
  ```bash
  bash install_conda.sh
  ```
  For visualization, we use our rasterizer that uses pytorch JIT Compiling Extensions. If there occurs a compiling error, you can install [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md) instead and set --rasterizer_type=pytorch3d when running the demos.

### Usage
1. Prepare data   
    download from huggingface data
2. run script: 
    ```bash
    python3 main.py
    ```