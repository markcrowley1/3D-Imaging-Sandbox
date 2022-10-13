"""
Description:
    Implementation of Canny Edge Detector.
    
    Edge Detector is composed of 5 steps:
        - Noise Reduction
        - Gradient Calculation
        - Non Maximum suppression
        - Double Threshold
        - Edge Tracking by Hysterersis

    Canny Edge detection is only effective on greyscale images.

    Credit to: https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123

Author:
    Mark Crowley
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpi

def load_imgs(dir_name: str):
    images = []

    for filename in os.listdir(dir_name):
        image = mpi.imread(dir_name + '/' + filename)
        images.append(image)

    return images

def gaussian_kernel(size, sigma = 1):
    """Create and return gaussian kernel"""
    pass

def reduce_noise():
    """Apply gaussian blur with gaussian kernel"""
    pass

def calculate_gradient():
    pass

def suppress_non_max():
    pass

def double_threshold():
    pass

def hysteresis():
    pass

def plot_img(imgs):
    plt.figure(figsize=(20, 40))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1,2,0)
        plt_idx = i+1
        plt.subplot(4, 2, plt_idx)    
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.show()

def main():
    # Load image
    img_directory = sys.argv[1]
    images = load_imgs(img_directory)
    # Plot images
    plot_img(images)


if __name__ == "__main__":
    main()