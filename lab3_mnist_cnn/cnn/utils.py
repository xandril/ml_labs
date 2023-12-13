import random

import numpy as np
from numba import njit

CLASS_NUMBER: int = 10
IMAGE_CHANNELS_COUNT: int = 1
IMAGE_SIZE: int = 28

Params = list[np.ndarray]

Outputs = list[np.ndarray]

Gradients = list[np.ndarray]

Moments = list[np.ndarray]

def create_filters(size: tuple[int, ...], scale: float = 1.0) -> np.ndarray:
    stddev: float = scale / np.sqrt(np.array(size).prod())
    return np.random.normal(loc=0, scale=stddev, size=size)


def initialize_weights(size: tuple[int, ...]) -> np.ndarray:
    return np.random.standard_normal(size=size) * 0.01


def shuffle(data: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices: list[int] = list(range(len(data)))
    random.shuffle(indices)
    return data[indices], target[indices]



@njit(cache=True)
def argmax2d(arr: np.ndarray) -> tuple[int, int]:
    idx: int = arr.argmax()
    return idx // arr.shape[1], idx % arr.shape[1]


@njit(cache=True)
def one_hot_coding(cls: int) -> np.ndarray:
    code: np.ndarray = np.zeros(shape=(CLASS_NUMBER, 1))
    code[cls][0] = 1.0
    return code


@njit(cache=True)
def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    return -(q * np.log(p)).sum()


def load_mnist() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from keras.datasets import mnist
    (train_data, train_target), (test_data, test_target) = mnist.load_data()

    def prepare(array: np.ndarray) -> np.ndarray:
        return array.astype(np.float32).reshape((len(array), IMAGE_CHANNELS_COUNT, IMAGE_SIZE, IMAGE_SIZE)) / 255

    return (prepare(train_data), train_target), (prepare(test_data), test_target)
