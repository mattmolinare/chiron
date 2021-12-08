import abc

import tensorflow as tf

__all__ = ["TriangularCLR", "Triangular2CLR", "ExponentialCLR"]


class CLR(tf.keras.callbacks.Callback, abc.ABC):
    """Cyclical learning rate (CLR) scheduler [1]_.

    References
    ----------
    .. [1] Smith, Leslie N. "Cyclical learning rates for training neural
        networks." 2017 IEEE Winter Conference on Applications of Computer
        Vision (WACV). IEEE, 2017.

    """

    def __init__(self, base_lr, max_lr, step_size):
        super().__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size

    def get_cycle(self, step):
        return int(1 + step / (2 * self.step_size))

    @abc.abstractmethod
    def get_scale(self, step, cycle):
        pass

    def get_lr(self, step, cycle):
        x = abs(step / self.step_size - 2 * cycle + 1)
        scale = self.get_scale(step, cycle)
        return (
            self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x) * scale
        )

    def on_train_begin(self, logs=None):
        self.step = 0

    def on_train_batch_begin(self, batch, logs=None):
        self.cycle = self.get_cycle(self.step)
        self.lr = self.get_lr(self.step, self.cycle)
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1


class TriangularCLR(CLR):
    """*triangular* CLR scheduler."""

    @staticmethod
    def get_scale(step, cycle):
        return 1


class Triangular2CLR(CLR):
    """*triangular2* CLR scheduler."""

    @staticmethod
    def get_scale(step, cycle):
        return 1 / (2 ** (cycle - 1))


class ExponentialCLR(CLR):
    """*exp_range* CLR scheduler."""

    def __init__(self, base_lr, max_lr, step_size, gamma):
        super().__init__(base_lr, max_lr, step_size)
        self.gamma = gamma

    def get_scale(self, step, cycle):
        return self.gamma ** step
