
# README for Image Captioning Project

## Overview
This project implements an image captioning system using TensorFlow and a deep learning model combining a CNN (for image feature extraction) and an LSTM (for generating captions). The goal is to generate descriptive captions for images automatically.

## Getting Started
### Prerequisites
- Python 3.x
- TensorFlow
- Pandas
- NumPy
- Scikit-learn

### Installation
1. Ensure Python and pip are installed.
2. Install required packages:
   ```bash
   pip install tensorflow pandas numpy scikit-learn
   ```

### Running the Experiments
To run the experiments:
1. Clone the repository to your local machine.
2. Place your dataset in the `archive/` directory, ensuring it follows the required format.
3. Run the script using Python:
   ```bash
   python image_captioning_script.py
   ```

## Code Attribution
### Copied Files
- None of the code files have been directly copied from external repositories.

### Modified Files
- The model architecture and data preprocessing are inspired by standard practices in image captioning tasks but have been significantly adapted and modified for this specific project with inspiration from `Caption Generation of Images using CNN and LSTM` by `Yousuf et. al`.

### Original Files
- All code provided in `FinalProject.py` is original work.

## Dataset Description
- The dataset used in this project consists of images and corresponding captions. We used the Flickr-8k dataset, which can be downloaded from [here](https://www.kaggle.com/adityajn105/flickr8k).
- The dataset should be in a folder named `archive/` with two subfolders: `Images/` containing the images and a file `captions.txt` containing the captions.
- Each line in `captions.txt` should correspond to an image in the `Images/` folder and contain the image file name and its associated caption.
- This dataset format is a standard in image captioning tasks and can be obtained from publicly available image-caption datasets.
- We provided the single image we used for generating an example caption titled `Dog.jpg`

## Notes
- The model is trained for 5 epochs, but this can be adjusted as needed.
- The script uses VGG16 for feature extraction, with weights pre-trained on ImageNet.
- The LSTM decoder is a custom TensorFlow model.
- The number of words (vocabulary size) in tokenizer is set to 30, which can be modified based on the dataset size and diversity.
