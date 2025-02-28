import cv2
import numpy as np

def generate_pyramids(image): 
    downsampled = [image]
    for _ in range(3):
        downsampled.append(cv2.pyrDown(downsampled[-1]))

    gaussian_pyramid = [image]
    for _ in range(3):
        gaussian_pyramid.append(cv2.GaussianBlur(gaussian_pyramid[-1], (5, 5), 1))
        gaussian_pyramid[-1] = cv2.pyrDown(gaussian_pyramid[-1])

    return downsampled, gaussian_pyramid

def run(image): 
    downsampled, gaussian_pyramid = generate_pyramids(image)
    return downsampled + gaussian_pyramid 
