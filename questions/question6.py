import cv2
import numpy as np

def apply_canny(image, threshold1, threshold2, aperture_size=3): 
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
    edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=aperture_size)
    return edges

def run(image): 
    edges_low = apply_canny(image, 50, 100, 3)
    edges_medium = apply_canny(image, 100, 200, 3)
    edges_high = apply_canny(image, 150, 250, 3)
    edges_large_aperture = apply_canny(image, 100, 200, 5)

    return [edges_low, edges_medium, edges_high, edges_large_aperture]
