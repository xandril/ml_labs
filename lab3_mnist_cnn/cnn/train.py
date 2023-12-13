import pickle
import sys

import numpy as np
from tqdm import tqdm

from .back_prop import back_prop
from .feed_forward import feed_forward
from .utils import (Gradients,
                    Params,
                    Moments,
                    create_filters,
                    cross_entropy,
                    initialize_weights,
                    load_mnist,
                    one_hot_coding,
                    shuffle)

FIRST_KERNEL_COUNT = 8
FIRST_KERNEL_CHANNEL = 1
FIRST_KERNEL_SIZE = 5
SECOND_KERNEL_COUNT = 8
SECOND_KERNEL_CHANNEL = FIRST_KERNEL_COUNT
SECOND_KERNEL_SIZE = 5
FLATTENED = 800
FIRST_DENSE_LAYER_SIZE = 150
SECOND_DENSE_LAYER_SIZE = 10


def train(lr: float, gamma: float, batch_size: int, parameters_file_name: str) -> None:
    (data, target), _ = load_mnist()
    shuffle(data, target)
    batches: list[tuple[np.ndarray, np.ndarray]] = [
        (data[i:i + batch_size], target[i:i + batch_size])
        for i in range(0, len(data), batch_size)
    ]
    filters_1: np.ndarray = create_filters(
        (FIRST_KERNEL_COUNT, FIRST_KERNEL_CHANNEL, FIRST_KERNEL_SIZE, FIRST_KERNEL_SIZE))
    filters_2: np.ndarray = create_filters(
        (SECOND_KERNEL_COUNT, FIRST_KERNEL_COUNT, SECOND_KERNEL_SIZE, SECOND_KERNEL_SIZE))
    dense_1: np.ndarray = initialize_weights((FIRST_DENSE_LAYER_SIZE, FLATTENED))
    dense_2: np.ndarray = initialize_weights((SECOND_DENSE_LAYER_SIZE, FIRST_DENSE_LAYER_SIZE))
    params: Params = [filters_1, filters_2, dense_1, dense_2]
    cost: list[float] = []
    print(f'learning rate:{lr}, Batch Size:{batch_size}, Nesterov\'s Gamma:{gamma}')
    moments: Moments = [np.zeros_like(filters_1), np.zeros_like(filters_2), np.zeros_like(dense_1),
                        np.zeros_like(dense_2)]

    progress = tqdm(batches, file=sys.stdout)
    for batch in progress:
        data, target = batch
        params, moments, current_cost = nesterov_gradient_descent(data, target, lr, gamma, params, moments)
        cost.append(current_cost)
        progress.set_description(f'Cost: {current_cost:.2f}')

    with open(parameters_file_name, 'wb') as parameters_file:
        pickle.dump([params, cost], parameters_file)


def nesterov_gradient_descent(batch_data: np.ndarray, batch_target: np.ndarray, lr: float, gamma: float, params: Params,
                              moments: Moments) -> (Params, Moments, float):
    cost: float = 0
    batch_size = len(batch_data)
    gradients: list[np.ndarray] = [np.zeros_like(params[i]) for i in range(len(params))]
    for i in range(batch_size):
        data: np.ndarray = batch_data[i]
        target: np.ndarray = one_hot_coding(batch_target[i])
        outputs = feed_forward(data, params)
        current_gradients: Gradients = back_prop(data, target, outputs, params)
        accum_gradients(gradients, current_gradients)
        cost += cross_entropy(outputs[-1], target)
    update(params, moments, gradients, gamma, lr, batch_size)

    cost = cost / batch_size

    return params, moments, cost


def update(params: Params, moments: Moments, gradients: Gradients, gamma: float, lr: float, batch_size: int) -> None:
    for i in range(len(params)):
        moments[i] = gamma * moments[i] + lr * gradients[i] / batch_size
        params[i] -= moments[i]


def accum_gradients(
        accumulator: Gradients,
        gradients: Gradients) -> None:
    for i in range(len(accumulator)):
        accumulator[i] += gradients[i]
