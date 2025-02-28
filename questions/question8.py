import cv2
import numpy as np

def resize_image(image): 
    height, width = image.shape[:2]

    resized_linear = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
    resized_bicubic = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)

    return [image, resized_linear, resized_bicubic]

def run(image): 
    return resize_image(image)
