"""带限角谱法（Matsushima 2009）+ 零填充。

角谱法（Angular Spectrum Method, ASM）是求解亥姆霍兹方程、计算光场自由空间传播的
一种频域方法。其核心思想是：
    1. 对输入光场做 FFT，得到角谱 A0(fx, fy)
    2. 乘以传递函数 H(fx, fy) = exp(i·kz·dz)
    3. 做 IFFT 得到输出光场 Uz

本模块实现带限版本的 ASM，包含：
    - 2 倍零填充（减小边界卷绕误差）
    - Matsushima 2009 带限滤波（防止远场混叠）
    - 倏逝波截断（物理上正确的衰减波过滤）

数据布局：
    所有数组形状为 (Nx, Ny)，complex128。
    行主序：axis 0 = y 方向（rows），axis 1 = x 方向（cols）。
    这与 numpy 默认约定一致：(row, col) = (y, x)。
"""
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

from lightprop.fft import fft2d, fftshift as _fftshift, ifftshift as _ifftshift


def propagate(
    U0: np.ndarray,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    lambda_nm: float,
    dz: float,
    padding: bool = True,
    band_limited: bool = True,
) -> np.ndarray:
    """使用角谱法将输入光场传播距离 dz。

    参数：
        U0: 输入光场，形状 (Nx, Ny)，complex128。
            这是在孔径平面（z=0）的光场分布。
        Nx, Ny: 网格尺寸。Nx 是行数（y 方向），Ny 是列数（x 方向）。
        dx, dy: 像素间距，单位 μm。
            dx 是 x 方向相邻像素的物理距离。
            dy 是 y 方向相邻像素的物理距离。
        lambda_nm: 波长，单位 nm。内部转换为 μm。
        dz: 传播距离，单位 mm。内部转换为 μm。
        padding: 是否启用 2 倍零填充。
            零填充将网格扩展到 (2Nx, 2Ny)，将原场放在中心，
            可以减小衍射波到达边界时的周期性卷绕（wrap-around）误差。
        band_limited: 是否启用 Matsushima 2009 带限滤波。
            带限滤波可以防止远场传播时，传递函数欠采样导致的混叠。

    返回：
        Uz: 输出光场，形状 (Nx, Ny)，complex128。
            这是在 z=dz 平面的光场分布。

    算法流程：
        1. 单位转换：nm -> μm, mm -> μm
        2. 如果启用零填充：
            a. 创建 (2Nx, 2Ny) 的零数组
            b. 将 U0 放在中心
            c. ifftshift -> FFT -> fftshift 得到频谱
            d. 构造频率坐标 fx, fy（fftshift 后零频在中心）
            e. 构造传递函数 H(fx, fy)
            f. 应用 H：频谱逐点相乘
            g. ifftshift -> IFFT -> fftshift 得到空间场
            h. 裁剪中心 (Nx, Ny) 区域作为输出
        3. 如果禁用零填充：
            a. 直接对 U0 做 FFT
            b. 构造频率坐标和传递函数
            c. 应用 H 后 IFFT
    """
    # 单位统一为 μm，避免计算中产生单位错误
    lambda_um = lambda_nm / 1000.0
    dz_um = dz * 1000.0

    if padding:
        # ========== 带零填充的路径 ==========
        # 零填充将网格扩大 2 倍，有效减小边界卷绕误差
        pnx = Nx * 2  # 填充后的行数
        pny = Ny * 2  # 填充后的列数
        Lx = Nx * dx  # 物理窗口 x 尺寸，单位 μm
        Ly = Ny * dy  # 物理窗口 y 尺寸，单位 μm

        # 创建填充后的零数组，并将原场 U0 放在中心
        pad = np.zeros((pnx, pny), dtype=np.complex128)
        ox = (pnx - Nx) // 2  # 行偏移
        oy = (pny - Ny) // 2  # 列偏移
        pad[ox:ox + Nx, oy:oy + Ny] = U0

        # 对填充后的场做 FFT
        # 顺序：ifftshift -> FFT -> fftshift
        # ifftshift 将零频从中心移到角落，符合 FFT 输入约定
        # fftshift 将零频移回中心，方便后续构造传递函数
        work = _ifftshift(pad)
        work = fft2d(work, inverse=False)
        work = _fftshift(work)

        # 构造频率坐标
        # 注意：fftshift 后的布局是零频在中心
        # 对于 fftshift 后的数组，索引 m/n 对应的频率为：
        #   f = (m - N/2) / (N * dx)   for m = 0, 1, ..., N-1
        k = 2.0 * np.pi / lambda_um
        half_pnx = pnx // 2
        half_pny = pny // 2

        # fy：随行索引（axis 0）变化，对应 y 方向的频率
        # fy 的范围是 [-1/(2*dy), 1/(2*dy)]，间隔 1/(pnx*dy)
        fy = (np.arange(pnx) - half_pnx) / (pnx * dy)   # shape (pnx,)
        # fx：随列索引（axis 1）变化，对应 x 方向的频率
        # fx 的范围是 [-1/(2*dx), 1/(2*dx)]，间隔 1/(pny*dx)
        fx = (np.arange(pny) - half_pny) / (pny * dx)   # shape (pny,)

        # 构建传递函数 H(fx, fy)
        # 使用 numpy 广播机制，避免显式双重循环
        # kx shape (1, pny)，ky shape (pnx, 1)，kxy2 shape (pnx, pny)
        # kx 对应 x 方向波矢，随列变化；ky 对应 y 方向波矢，随行变化
        kx = 2.0 * np.pi * fx[None, :]   # (1, pny)
        ky = 2.0 * np.pi * fy[:, None]   # (pnx, 1)
        kxy2 = kx ** 2 + ky ** 2

        # 初始化传递函数为 0（倏逝波默认被滤除）
        H = np.zeros((pnx, pny), dtype=np.complex128)

        # 传播模式判定：kxy2 <= k^2 表示该频率分量可以传播（非倏逝波）
        # kxy2 > k^2 对应倏逝波，指数衰减，不参与远场传播
        propagating = kxy2 <= k * k
        if np.any(propagating):
            kz = np.sqrt(k * k - kxy2)  # z 方向波矢
            phase = kz * dz_um
            H[propagating] = np.exp(1j * phase[propagating])

        # Matsushima 2009 带限滤波
        # 问题：传递函数 H(fx, fy) 在离散频率点采样时，
        #       在远场（dz 很大）会变化非常剧烈，导致欠采样和混叠。
        # 解决：在频域施加一个低通滤波器，只保留低频分量。
        #
        # 截止频率公式：
        #   f_limit = 1/(2·dx) * 1/sqrt(1 + (λ·dz/L)^2)
        #
        # 物理直觉：
        #   - dz -> 0（近场）：f_limit -> 1/(2*dx)，不切任何频率
        #   - dz -> ∞（远场）：f_limit -> 0，几乎所有频率被滤掉
        #   - 波长越长、传播越远，能正确传播的最高频率越低
        if band_limited:
            fx_max = 1.0 / (2.0 * dx)  # x 方向奈奎斯特频率
            fy_max = 1.0 / (2.0 * dy)  # y 方向奈奎斯特频率
            # 带限截止频率
            fx_limit = fx_max / np.sqrt(1.0 + (lambda_um * dz_um / Lx) ** 2)
            fy_limit = fy_max / np.sqrt(1.0 + (lambda_um * dz_um / Ly) ** 2)

            # 构造带限掩膜：同时满足 x/y 方向截止频率且为传播模式
            band_mask = (np.abs(fx[None, :]) <= fx_limit) & (np.abs(fy[:, None]) <= fy_limit)
            H = np.where(band_mask & propagating, H, np.complex128(0))

        # 在频域应用传递函数：逐点复数乘法
        work = work * H

        # 逆 FFT 回到空间域
        # 顺序：ifftshift -> IFFT -> fftshift
        work = _ifftshift(work)
        work = fft2d(work, inverse=True)  # scipy ifft2 已包含 1/(Nx*Ny) 归一化
        work = _fftshift(work)

        # 从填充后的网格中裁剪中心区域，得到原始尺寸的输出
        Uz = work[ox:ox + Nx, oy:oy + Ny].copy()

    else:
        # ========== 无零填充的基础路径 ==========
        # 适用于近场或网格足够大的情况
        work = U0.copy()
        work = _ifftshift(work)
        work = fft2d(work, inverse=False)
        work = _fftshift(work)

        Lx = Nx * dx
        Ly = Ny * dy
        k = 2.0 * np.pi / lambda_um
        half_nx = Nx // 2
        half_ny = Ny // 2

        # 频率坐标（无填充，尺寸为 Nx x Ny）
        fy = (np.arange(Nx) - half_nx) / (Nx * dy)   # (Nx,)
        fx = (np.arange(Ny) - half_ny) / (Ny * dx)   # (Ny,)

        kx = 2.0 * np.pi * fx[None, :]
        ky = 2.0 * np.pi * fy[:, None]
        kxy2 = kx ** 2 + ky ** 2

        H = np.zeros((Nx, Ny), dtype=np.complex128)
        propagating = kxy2 <= k * k
        if np.any(propagating):
            kz = np.sqrt(k * k - kxy2)
            phase = kz * dz_um
            H[propagating] = np.exp(1j * phase[propagating])

        if band_limited:
            fx_max = 1.0 / (2.0 * dx)
            fy_max = 1.0 / (2.0 * dy)
            fx_limit = fx_max / np.sqrt(1.0 + (lambda_um * dz_um / Lx) ** 2)
            fy_limit = fy_max / np.sqrt(1.0 + (lambda_um * dz_um / Ly) ** 2)

            band_mask = (np.abs(fx[None, :]) <= fx_limit) & (np.abs(fy[:, None]) <= fy_limit)
            H = np.where(band_mask & propagating, H, np.complex128(0))

        work = work * H

        work = _ifftshift(work)
        work = fft2d(work, inverse=True)
        work = _fftshift(work)
        Uz = work.copy()

    return Uz
