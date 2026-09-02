"""二维 FFT 工具模块。

本模块基于 scipy.fft 实现，后端为 FFTW 或 MKL，通常比纯 Python/JS 实现快 10~100 倍。
数据布局：numpy ndarray，形状 (Nx, Ny)，complex128，行主序（row=y, col=x）。
"""
import numpy as np
from scipy.fft import fft2 as _scipy_fft2, ifft2 as _scipy_ifft2
from scipy.fft import fftshift as _scipy_fftshift, ifftshift as _scipy_ifftshift


def fft2d(arr: np.ndarray, inverse: bool = False) -> np.ndarray:
    """二维 FFT/IFFT（返回新数组，不修改输入）。

    参数：
        arr: 输入数组，形状 (Nx, Ny)，complex128
        inverse: 如果为 True，执行逆 FFT 并自动归一化（scipy ifft2 已包含 1/(Nx*Ny)）

    返回：
        变换后的数组，形状 (Nx, Ny)，complex128
    """
    if inverse:
        return _scipy_ifft2(arr)
    return _scipy_fft2(arr)


def fftshift(arr: np.ndarray) -> np.ndarray:
    """fftshift：将零频分量从角落移到中心。

    在角谱法中，FFT 后的频谱零频在角落，需要移到中心以便：
    1. 可视化显示
    2. 构造以中心为对称轴的带限掩膜
    3. 物理上更直观的频率坐标对应

    参数：
        arr: 输入数组，形状 (Nx, Ny)

    返回：
        移位后的数组
    """
    return _scipy_fftshift(arr)


def ifftshift(arr: np.ndarray) -> np.ndarray:
    """ifftshift：将零频分量从中心移回角落。

    在执行 FFT 前，需要将频谱从中心移回角落，以符合 FFT 算法的输入约定。

    参数：
        arr: 输入数组，形状 (Nx, Ny)

    返回：
        移位后的数组
    """
    return _scipy_ifftshift(arr)


def fft_self_test() -> dict:
    """FFT 自检：验证变换的正确性。

    测试内容：
    1. 1D 脉冲响应：delta(t) 的 FFT 应该是常数
    2. 2D 随机数组 roundtrip：FFT -> IFFT 应恢复原值
    3. fftshift/ifftshift roundtrip：奇尺寸和偶尺寸

    返回：
        字典，包含各项测试的最大误差
    """
    results = {}

    # 1D 脉冲测试
    # 期望：脉冲的 FFT 是全为 1 的常数
    n = 8
    arr = np.zeros(n, dtype=np.complex128)
    arr[0] = 1.0
    fwd = fft2d(arr.reshape(1, -1))  # shape (1, 8)
    expected = np.ones_like(fwd)
    results["1d_impulse_max_err"] = float(np.max(np.abs(fwd - expected)))

    # 2D 随机 roundtrip 测试
    # 期望：FFT -> IFFT 后恢复原值，误差接近机器精度
    nx, ny = 16, 16
    orig = np.random.randn(nx, ny) + 1j * np.random.randn(nx, ny)
    orig = orig.astype(np.complex128)
    fwd = fft2d(orig)
    inv = fft2d(fwd, inverse=True)
    results["2d_roundtrip_max_err"] = float(np.max(np.abs(inv - orig)))

    # fftshift roundtrip（奇数尺寸）
    nx3, ny3 = 5, 5
    c = np.random.randn(nx3, ny3) + 1j * np.random.randn(nx3, ny3)
    c = c.astype(np.complex128)
    orig3 = c.copy()
    c = fftshift(c)
    c = ifftshift(c)
    results["fftshift_odd_max_err"] = float(np.max(np.abs(c - orig3)))

    # fftshift roundtrip（偶数尺寸）
    nx4, ny4 = 4, 4
    d = np.random.randn(nx4, ny4) + 1j * np.random.randn(nx4, ny4)
    d = d.astype(np.complex128)
    orig4 = d.copy()
    d = fftshift(d)
    d = ifftshift(d)
    results["fftshift_even_max_err"] = float(np.max(np.abs(d - orig4)))

    return results
