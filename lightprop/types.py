"""数据结构定义。

本模块定义了角谱法传播计算所需的全部参数和输出结构。
所有物理量的单位都在字段注释中标注，使用时务必注意单位统一。
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Params:
    """单次传播计算的全部参数。

    属性说明：
        aperture_type: 光阑类型，可选值：
            - "slit": 单缝
            - "doubleSlit": 双缝
            - "circle": 圆孔
            - "rectangle": 矩形孔
            - "free": 自由空间（全 apertured）
            - "upload": 上传图片（由前端预处理好像素）
        aperture_params: 光阑参数字典，根据 aperture_type 不同而不同：
            - slit: {"width": 缝宽，单位 μm}
            - doubleSlit: {"width": 缝宽, "separation": 双缝间距，单位 μm}
            - circle: {"radius": 半径，单位 μm}
            - rectangle: {"width": 宽, "height": 高，单位 μm}
        Nx, Ny: 网格尺寸（行数，列数）。注意：Nx 对应 y 方向，Ny 对应 x 方向。
        dx, dy: 像素间距，单位 μm。dx 是 x 方向的间隔，dy 是 y 方向的间隔。
        lambda_nm: 波长，单位 nm。内部会转换为 μm 使用。
        dz: 传播距离，单位 mm。内部会转换为 μm 使用。
        w0: 高斯包络的束腰半径，单位 mm。
            当 w0 >= 100 时视为平面波，不施加高斯包络。
        tilt_on: 是否开启偏轴照明。
        tilt_x_deg, tilt_y_deg: 偏轴角度，单位度。仅在 tilt_on=True 时生效。
        temporal_on: 是否开启时间相干性（宽光谱）。
        delta_lambda: 光谱宽度 FWHM，单位 nm。
        M: 时间相干性采样波长数。
        spatial_on: 是否开启空间相干性（扩展光源）。
        K: 空间相干性采样方向数。
        padding: 是否启用 2 倍零填充。零填充可以减小衍射波到达边界时的卷绕误差。
        band_limited: 是否启用 Matsushima 2009 带限滤波。带限可以防止远场传播时的高频混叠。
    """
    aperture_type: str = "slit"
    aperture_params: dict = field(default_factory=dict)
    Nx: int = 256
    Ny: int = 256
    dx: float = 10.0       # μm
    dy: float = 10.0       # μm
    lambda_nm: float = 532.0  # nm
    dz: float = 100.0      # mm
    w0: float = 100.0      # mm, Gaussian envelope; >=100 treated as plane wave
    tilt_on: bool = False
    tilt_x_deg: float = 0.0
    tilt_y_deg: float = 0.0
    temporal_on: bool = False
    delta_lambda: float = 20.0  # nm
    M: int = 5             # wavelength samples
    spatial_on: bool = False
    K: int = 5             # direction samples
    padding: bool = True
    band_limited: bool = True


@dataclass
class Result:
    """单次传播计算的输出结果。

    属性说明：
        intensity: 强度图，形状 (Ny, Nx)，uint8 类型，范围 [0, 255]。
            归一化方式：sqrt(I / I_max) * 255，这是为了在显示时更好地看到弱信号。
        phase: 相位图，形状 (Ny, Nx)，uint8 类型，范围 [0, 255]。
            映射方式：[-π, π] -> [0, 255]。
            在强度很弱的区域，相位会被设为透明（alpha=0）。
        cross_x: 沿 x 方向的中心截面强度，形状 (Ny,)。
            对应于 intensity[:, cy]，其中 cy = Ny // 2。
            表示 y=中心行时，强度随 x 的变化。
        cross_y: 沿 y 方向的中心截面强度，形状 (Nx,)。
            对应于 intensity[cx, :]，其中 cx = Nx // 2。
            表示 x=中心列时，强度随 y 的变化。
        info: 元数据字典，包含：
            - fresnelNum: 菲涅尔数 F = (Lx/2)^2 / (λ·dz)
            - Lx, Ly: 物理窗口尺寸，单位 mm
            - nx, ny: 实际使用的网格尺寸
            - computationTime: 计算耗时，单位 ms
            - samplingOk: 采样是否充足
            - maxInt: 最大强度值（归一化前）
    """
    intensity: np.ndarray   # (Ny, Nx) uint8, [0, 255]
    phase: np.ndarray       # (Ny, Nx) uint8, [0, 255]
    cross_x: np.ndarray     # (Ny,) float64 — intensity along x-axis center
    cross_y: np.ndarray     # (Nx,) float64 — intensity along y-axis center
    info: dict = field(default_factory=dict)
