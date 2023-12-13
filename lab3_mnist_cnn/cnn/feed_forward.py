from numba import njit
import numpy as np

from .utils import Params, Outputs


@njit(cache=True)
def convolution(image: np.ndarray, kernels: np.ndarray, stride: int) -> np.ndarray:
    kernels_count, kernels_channels, kernels_size, _ = kernels.shape
    image_channels, image_size, _ = image.shape

    feature_map_size: int = (image_size - kernels_size) // stride + 1
    feature_maps: np.ndarray = np.zeros(shape=(kernels_count, feature_map_size, feature_map_size))

    for k_ind in range(kernels_count):
        kernel: np.ndarray = kernels[k_ind]
        feature_map: np.ndarray = feature_maps[k_ind]

        y: int = 0
        for offset_y in range(0, image_size - kernels_size + 1, stride):
            x: int = 0
            for offset_x in range(0, image_size - kernels_size + 1, stride):
                feature_map[y][x] = \
                    (kernel * image[:, offset_y:offset_y + kernels_size, offset_x:offset_x + kernels_size]).sum()
                x += 1
            y += 1

    return feature_maps


@njit(cache=True)
def max_pool(image: np.ndarray, patch_size: int, stride: int):
    image_channels, image_size, _ = image.shape
    pooled_size: int = (image_size - patch_size) // stride + 1
    pooled_img: np.ndarray = np.zeros(shape=(image_channels, pooled_size, pooled_size))

    for channel_index in range(image_channels):
        channel: np.ndarray = image[channel_index]
        pooled_channel: np.ndarray = pooled_img[channel_index]

        y: int = 0
        for offset_y in range(0, image_size - patch_size + 1, stride):
            x: int = 0
            for offset_x in range(0, image_size - patch_size + 1, stride):
                pooled_channel[y, x] = \
                    channel[offset_y:offset_y + patch_size, offset_x:offset_x + patch_size].max()
                x += 1
            y += 1

    return pooled_img


@njit(cache=True)
def soft_max(z: np.ndarray) -> np.ndarray:
    exp: np.ndarray = np.exp(z)
    return exp / exp.sum()


def feed_forward(image: np.ndarray, params: Params) -> Outputs:
    kernels1, kernels2, dense1, dense2 = params

    conv1_out: np.ndarray = convolution(image, kernels1, stride=1)
    conv1_out[conv1_out <= 0] = 0

    conv2_out: np.ndarray = convolution(conv1_out, kernels2, stride=1)
    conv2_out[conv2_out <= 0] = 0
    pooled: np.ndarray = max_pool(conv2_out, patch_size=2, stride=2)

    maps_count, map_size, _ = pooled.shape
    flattened: np.ndarray = pooled.reshape((maps_count * map_size * map_size, 1))

    dense1_out: np.ndarray = dense1.dot(flattened)
    dense1_out[dense1_out <= 0] = 0

    dense2_out: np.ndarray = dense2.dot(dense1_out)
    result: np.ndarray = soft_max(dense2_out)

    return [conv1_out, conv2_out, pooled, flattened, dense1_out, result]


def classify(image: np.ndarray, params: Params) -> (int, np.ndarray):
    ff_out = feed_forward(image, params)[-1]
    return ff_out.argmax(), ff_out
