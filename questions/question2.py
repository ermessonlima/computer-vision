import numpy as np
import streamlit as st

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
 
    kernel_size = st.slider("Tamanho do Kernel", min_value=3, max_value=7, step=2, value=3)
 
    kernels = {
        3: np.array([[-1, -1, -1],
                     [-1,  8, -1],
                     [-1, -1, -1]]),

        5: np.array([[-1, -1, -1, -1, -1],
                     [-1,  1,  2,  1, -1],
                     [-1,  2,  4,  2, -1],
                     [-1,  1,  2,  1, -1],
                     [-1, -1, -1, -1, -1]]),

        7: np.array([[-1, -1, -1, -1, -1, -1, -1],
                     [-1,  1,  1,  2,  1,  1, -1],
                     [-1,  1,  2,  3,  2,  1, -1],
                     [-1,  2,  3,  4,  3,  2, -1],
                     [-1,  1,  2,  3,  2,  1, -1],
                     [-1,  1,  1,  2,  1,  1, -1],
                     [-1, -1, -1, -1, -1, -1, -1]])
    }
 
    filtered_image = apply_convolution(image, kernels[kernel_size])

    return [filtered_image]
