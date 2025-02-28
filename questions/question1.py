import numpy as np

def apply_convolution(image, kernel):
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2
    height, width, _ = image.shape

    output = np.zeros_like(image)

    for c in range(3):  
        for i in range(pad, height - pad):
            for j in range(pad, width - pad):
                region = image[i-pad:i+pad+1, j-pad:j+pad+1, c]
                output[i, j, c] = np.clip(np.sum(region * kernel), 0, 255)

    return output

def run(image): 
    kernel_mean = np.ones((3, 3)) / 9  
    kernel_sharpen = np.array([[ 0, -1,  0],
                                [-1,  5, -1],
                                [ 0, -1,  0]])   

    filtered_image_mean = apply_convolution(image, kernel_mean)
    filtered_image_sharpen = apply_convolution(image, kernel_sharpen)

    return [filtered_image_mean, filtered_image_sharpen]
