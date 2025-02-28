import numpy as np

def bilateral_filter(image, d=5, sigma_color=75, sigma_space=75): 
    height, width, _ = image.shape
    output = np.copy(image)
    pad = d // 2

    spatial_kernel = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            spatial_kernel[i, j] = np.exp(-((i - pad)**2 + (j - pad)**2) / (2 * sigma_space**2))

    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            for c in range(3):   
                neighborhood = image[i-pad:i+pad+1, j-pad:j+pad+1, c]
                color_differences = (neighborhood - image[i, j, c])**2
                color_weights = np.exp(-color_differences / (2 * sigma_color**2))
                weights = color_weights * spatial_kernel
                output[i, j, c] = np.sum(neighborhood * weights) / np.sum(weights)

    return output

def run(image): 
    filtered_bilateral = bilateral_filter(image)
    return [filtered_bilateral]
