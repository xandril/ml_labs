from numba import njit
import numpy as np

from .utils import Params, Gradients, Outputs
from .utils import argmax2d



@njit(cache=True)
def conv_back_prop(d_conv_prev: np.ndarray, image: np.ndarray, kernels: np.ndarray, stride: int) -> tuple[
    np.ndarray, np.ndarray]:
    kernels_count, kernels_channels, kernel_size, _ = kernels.shape
    _, image_size, _ = image.shape

    gradients: np.ndarray = np.zeros_like(image)
    d_kernels: np.ndarray = np.zeros_like(kernels)

    for k_ind in range(kernels_count):
        kernel: np.ndarray = kernels[k_ind]
        d_kernel: np.ndarray = d_kernels[k_ind]
        d_conv: np.ndarray = d_conv_prev[k_ind]

        y: int = 0
        for offset_y in range(0, image_size - kernel_size + 1, stride):
            x: int = 0
            for offset_x in range(0, image_size - kernel_size + 1, stride):
                d_kernel += d_conv[y, x] * image[:, offset_y:offset_y + kernel_size, offset_x:offset_x + kernel_size]
                gradients[:, offset_y:offset_y + kernel_size, offset_x:offset_x + kernel_size] += d_conv[y, x] * kernel
                x += 1
            y += 1

    return gradients, d_kernels


@njit(cache=True)
def max_pool_grad(d_pooled: np.ndarray, image: np.ndarray, patch_size: int, stride: int) -> np.ndarray:
    channels_count, image_size, _ = image.shape

    gradients: np.ndarray = np.zeros_like(image)

    for channel_index in range(channels_count):
        channel: np.ndarray = image[channel_index]
        channel_gradient: np.ndarray = gradients[channel_index]
        d_pooled: np.ndarray = d_pooled[channel_index]

        out_y: int = 0
        for curr_y in range(0, image_size - patch_size + 1, stride):
            out_x: int = 0
            for curr_x in range(0, image_size - patch_size + 1, stride):
                (y, x) = argmax2d(channel[curr_y:curr_y + patch_size, curr_x:curr_x + patch_size])
                channel_gradient[curr_y + y, curr_x + x] = d_pooled[out_y, out_x]
                out_x += 1
            out_y += 1

    return gradients


def back_prop(image: np.ndarray, target: np.ndarray, outputs: Outputs, params: Params) -> Gradients:
    kernels1, kernels2, dense1, dense2 = params
    conv1_out, conv2_out, pooled, flattened, dense1_out, result = outputs

    d_loss: np.ndarray = result - target
    d_dense2: np.ndarray = d_loss.dot(dense1_out.T)

    d_dense1_out: np.ndarray = dense2.T.dot(d_loss)
    d_dense1_out[dense1_out <= 0] = 0
    d_dense1: np.ndarray = d_dense1_out.dot(flattened.T)

    d_flattened = dense1.T.dot(d_dense1_out)
    d_pooled: np.ndarray = d_flattened.reshape(pooled.shape)

    d_conv2_out: np.ndarray = max_pool_grad(d_pooled, conv2_out, patch_size=2, stride=2)
    d_conv2_out[conv2_out <= 0] = 0

    d_conv1_out, d_conv2 = conv_back_prop(d_conv2_out, conv1_out, kernels2, stride=1)
    d_conv1_out[conv1_out <= 0] = 0

    _, d_conv1 = conv_back_prop(d_conv1_out, image, kernels1, stride=1)

    grads = [d_conv1, d_conv2, d_dense1, d_dense2]

    return grads
