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

def sobel_filter(image): 
    sobel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]])

    sobel_x_result = apply_convolution(image, sobel_x)
    sobel_y_result = apply_convolution(image, sobel_y)

    magnitude = np.sqrt(sobel_x_result.astype(np.float64)**2 + sobel_y_result.astype(np.float64)**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

    return [sobel_x_result, sobel_y_result, magnitude]

def run(image): 
    return sobel_filter(image)
