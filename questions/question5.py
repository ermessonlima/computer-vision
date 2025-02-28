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

def sobel_and_gaussian(image): 

    gaussian_kernel = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ]) / 273

    sobel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]])
 
    sobel_x_result = apply_convolution(image, sobel_x)
    sobel_y_result = apply_convolution(image, sobel_y)
    magnitude_sobel = np.sqrt(sobel_x_result.astype(np.float64)**2 + sobel_y_result.astype(np.float64)**2)
    magnitude_sobel = np.clip(magnitude_sobel, 0, 255).astype(np.uint8)
 
    gaussian_blurred = apply_convolution(image, gaussian_kernel)
    sobel_x_gaussian = apply_convolution(gaussian_blurred, sobel_x)
    sobel_y_gaussian = apply_convolution(gaussian_blurred, sobel_y)
    magnitude_gaussian_sobel = np.sqrt(sobel_x_gaussian.astype(np.float64)**2 + sobel_y_gaussian.astype(np.float64)**2)
    magnitude_gaussian_sobel = np.clip(magnitude_gaussian_sobel, 0, 255).astype(np.uint8)

    return [magnitude_sobel, magnitude_gaussian_sobel]

def run(image): 
    return sobel_and_gaussian(image)
