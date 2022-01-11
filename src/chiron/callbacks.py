import abc
import functools

import numpy as np
import tensorflow as tf


class DecoratorMeta(type):

    method_names = [
        "on_epoch_begin",
        "on_epoch_end",
        "on_predict_batch_begin",
        "on_predict_batch_end",
        "on_predict_begin",
        "on_predict_end",
        "on_test_batch_begin",
        "on_test_batch_end",
        "on_test_begin",
        "on_test_end",
        "on_train_batch_begin",
        "on_train_batch_end",
        "on_train_begin",
        "on_train_end",
    ]

    def __new__(cls, name, bases, attrs):
        callback = super().__new__(cls, name, bases, attrs)
        decorator = callback.decorator
        for method_name in cls.method_names:
            method = getattr(callback, method_name, None)
            if method is not None:
                setattr(callback, method_name, decorator(method))
        return callback


class AbstractDecoratorMeta(DecoratorMeta, abc.ABCMeta):
    pass


class DecoratorMixin(metaclass=AbstractDecoratorMeta):
    @classmethod
    @abc.abstractmethod
    def decorator(cls, method):
        pass


class CustomLogsMixin(DecoratorMixin):
    @staticmethod
    def unpack_logs(args, kwargs):
        if "logs" in kwargs:
            logs = kwargs["logs"]
        else:
            *args, logs = args
        return args, logs

    @classmethod
    def decorator(cls, method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            args, logs = cls.unpack_logs(args, kwargs)
            logs = logs or {}
            logs = self.update_logs(logs)
            method(self, *args, logs=logs)

        return wrapper

    @abc.abstractmethod
    def update_logs(self, logs):
        pass


class LearningRateMixin(CustomLogsMixin):
    def get_lr(self):
        return tf.keras.backend.eval(self.model.optimizer.lr)

    def update_logs(self, logs):
        lr = logs.get("lr")
        if lr is None:
            lr = self.get_lr()
        return {"lr": lr}


class NamedLogsMixin(CustomLogsMixin):
    def __init__(self, include_names=None, exclude_names=None, **kwargs):
        super().__init__(**kwargs)
        self.include_names = include_names
        self.exclude_names = exclude_names

    def update_logs(self, logs):
        if self.include_names is not None:
            return {
                name: logs[name] for name in self.include_names if name in logs
            }
        elif self.exclude_names is not None:
            return {
                name: value
                for name, value in logs.items()
                if name not in self.exclude_names
            }
        else:
            return logs


class TensorBoardLearningRate(
    LearningRateMixin, tf.keras.callbacks.TensorBoard
):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            {
                "histogram_freq": 0,
                "write_images": False,
                "profile_batch": 0,
                "embeddings_freq": 0,
                "embeddings_metadata": None,
            }
        )
        super().__init__(*args, **kwargs)


class TensorBoardNamedLogs(NamedLogsMixin, tf.keras.callbacks.TensorBoard):
    pass


class ProgbarLoggerNamedLogs(NamedLogsMixin, tf.keras.callbacks.ProgbarLogger):
    pass


class SilentTerminateOnNaN(tf.keras.callbacks.Callback):
    """Silent version of :class:`tf.keras.callbacks.TerminateOnNaN`.

    Allows user to specify which quantity to monitor.
    """

    def __init__(self, monitor="loss"):
        super().__init__()
        self.monitor = monitor

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        value = logs.get(self.monitor)
        if value is not None:
            if np.isnan(value) or np.isinf(value):
                self.model.stop_training = True


class LearningRateFinder(tf.keras.callbacks.Callback):
    """Learning rate finder [1]_.

    References
    ----------
    .. [1] https://docs.fast.ai/callbacks.lr_finder.html

    """

    def __init__(
        self,
        monitor="loss",
        base_lr=1e-7,
        max_lr=10.0,
        max_steps=1000,
        smoothing=0.9,
        mode="auto",
    ):
        super().__init__()
        self.monitor = monitor
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.max_steps = max_steps
        self.smoothing = smoothing
        self.mode = mode
        self.minimize = (
            self.mode == "min"
            or self.mode == "auto"
            and "loss" in self.monitor
        )
        self.monitor_op = np.less if self.minimize else np.greater
        self.step = None
        self.lr = None
        self.mean_value = None
        self.best_value = None
        self.history = None

    def get_lr(self, step):
        if step > self.max_steps:
            return self.max_lr
        else:
            return self.base_lr * (self.max_lr / self.base_lr) ** (
                step / self.max_steps
            )

    def on_train_begin(self, logs=None):
        self.step = 0
        self.mean_value = 0.0
        self.best_value = np.inf if self.minimize else -np.inf
        self.history = {"lr": [], self.monitor: []}

    def on_train_batch_begin(self, batch, logs=None):
        self.lr = self.get_lr(self.step)
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        value = logs.get(self.monitor)
        if value is not None:
            if self.monitor_op(value, self.best_value):
                self.best_value = value
            self.mean_value = (
                self.smoothing * self.mean_value + (1 - self.smoothing) * value
            )
            smoothed_value = self.mean_value / (
                1 - self.smoothing ** (self.step + 1)
            )
            if self.monitor_op(4 * self.best_value, smoothed_value):
                self.model.stop_training = True
            self.history["lr"].append(self.lr)
            self.history[self.monitor].append(value)
        if self.step == self.max_steps:
            self.model.stop_training = True
        self.step += 1


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
        self.step = None
        self.cycle = None
        self.lr = None

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
