from .feed_forward import classify
from .train import train
from .utils import Params, load_mnist, CLASS_NUMBER

__all__ = ['train', 'Params', 'classify', 'load_mnist', 'CLASS_NUMBER']
