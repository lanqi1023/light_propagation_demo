import numpy as np
import matplotlib.pyplot as plt

DELTA = 1e-15 # 1fm
SAMPLE_PERIOD = 1e-9 # 1nm
WIDTH = 1e-6 # 1um

class continuous_signal:
    def __init__(self, signal: np.ndarray):
        self.signal = signal

    def integral(self) -> float:
        return np.sum(self.signal) * DELTA

    def sample(self) -> discrete_signal:
        return discrete_signal(
            self.signal[::int(SAMPLE_PERIOD / DELTA)]
        )
    
    def plot(self, axes: plt.Axes = plt.gca()):
        axes.plot(self.sample().signal)

class continuous_2d_signal:
    def __init__(self, signal: np.ndarray):
        self.signal = signal

    def integral(self) -> float:
        return np.sum(self.signal) * DELTA * DELTA

    def sample(self) -> discrete_2d_signal:
        return discrete_2d_signal(
            self.signal[::int(SAMPLE_PERIOD / DELTA), ::int(SAMPLE_PERIOD / DELTA)]
        )
    
    def plot(self, axes: plt.Axes = plt.gca()):
        axes.imshow(self.sample().signal)

class discrete_signal:
    def __init__(self, signal: np.ndarray):
        self.signal = signal

class discrete_2d_signal:
    def __init__(self, signal: np.ndarray):
        self.signal = signal

signal = continuous_2d_signal(
    np.exp(-((np.arange(-WIDTH/2, WIDTH/2, DELTA)[:, None])**2 + (np.arange(-WIDTH/2, WIDTH/2, DELTA)[None, :])**2) / (2 * (WIDTH/10)**2))
)

signal.plot()