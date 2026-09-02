"""非理想光学效应：偏轴照明、时间相干性、空间相干性。

在实际光学系统中，理想单色点光源的假设往往不成立。本模块实现三种常见的非理想效应：

1. 偏轴照明（Tilt）：入射光不是正入射，而是以一个倾斜角度入射。
   这会在孔径平面上引入一个线性相位因子，导致远场图样平移或形变。

2. 时间相干性（Temporal Coherence）：光源不是单色光，而是有一定的光谱宽度 Δλ。
   不同波长的光传播后产生的图样不同，需要将多个波长的强度图样加权求和。
   这会导致干涉条纹的对比度下降。

3. 空间相干性（Spatial Coherence）：光源不是点光源，而是有一定尺寸的扩展光源。
   可以看作从光源不同点发出的光以不同角度入射。
   这也会导致干涉条纹的对比度下降，但物理机制与时间相干性不同。

所有函数都接受形状 (Nx, Ny) 的 numpy 数组，complex128 类型。
"""
import numpy as np


def apply_tilt(U: np.ndarray, Nx: int, Ny: int, dx: float, dy: float,
                lambda_nm: float, theta_x_deg: float, theta_y_deg: float) -> np.ndarray:
    """施加偏轴照明相位：U *= exp(i * k * (sinθx * x + sinθy * y))

    物理意义：当入射光以角度 θx, θy 入射时，在孔径平面上引入一个线性相位调制。
    这相当于在远场（傅里叶平面）将图样平移。

    参数：
        U: 输入光场，形状 (Nx, Ny)，complex128。会被原地修改并返回。
        Nx, Ny: 网格尺寸
        dx, dy: 像素间距，单位 μm
        lambda_nm: 波长，单位 nm
        theta_x_deg: x 方向入射角，单位度
        theta_y_deg: y 方向入射角，单位度

    返回：
        修改后的光场 U，形状 (Nx, Ny)，complex128
    """
    # 如果角度都为 0，直接返回（正入射，无调制）
    if not theta_x_deg and not theta_y_deg:
        return U

    # 单位转换和波数计算
    lambda_um = lambda_nm / 1000.0  # nm -> μm
    k = 2.0 * np.pi / lambda_um     # 波数，单位 μm^-1

    # 角度转弧度，计算 sinθ
    theta_x = np.radians(theta_x_deg)
    theta_y = np.radians(theta_y_deg)
    sin_tx = np.sin(theta_x)
    sin_ty = np.sin(theta_y)

    # 构造空间坐标
    # x 坐标随列索引变化，y 坐标随行索引变化
    xc = (Ny - 1) / 2.0
    yc = (Nx - 1) / 2.0
    x = (np.arange(Ny) - xc) * dx   # (Ny,)，x 方向坐标
    y = (np.arange(Nx) - yc) * dy   # (Nx,)，y 方向坐标

    # 相位 = k * (sinθx * x + sinθy * y)
    # 使用广播机制，phase shape = (Nx, Ny)
    phase = k * (sin_tx * x[None, :] + sin_ty * y[:, None])
    U *= np.exp(1j * phase)
    return U


def temporal_coherence(
    U0_template: np.ndarray,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    lambda0_nm: float,
    delta_lambda: float,
    M: int,
    dz: float,
    padding: bool,
    band_limited: bool,
) -> np.ndarray:
    """时间相干性：多波长非相干求和。

    物理模型：光源有有限的光谱宽度 Δλ（FWHM），可以看作是多个波长的叠加。
    每个波长独立传播，最终的强度图样是各波长强度图样的加权和：
        I_total = Σ w_m * |U(λ_m)|^2

    权重 w_m 是高斯分布，FWHM = Δλ。

    参数：
        U0_template: 孔径光场模板，形状 (Nx, Ny)，complex128。
            该模板不随波长改变，用于在每个波长下重新生成光场。
        Nx, Ny: 网格尺寸
        dx, dy: 像素间距，单位 μm
        lambda0_nm: 中心波长，单位 nm
        delta_lambda: 光谱宽度 FWHM，单位 nm
        M: 采样波长数。通常取 3, 5, 7。
            计算量与 M 成正比，M 太大会导致计算缓慢。
        dz: 传播距离，单位 mm
        padding: 是否零填充
        band_limited: 是否带限

    返回：
        intensity: 总强度图，形状 (Nx, Ny)，float64。
            如果 M <= 1，返回 None（单色光无需此计算）。
    """
    from lightprop.angspec import propagate

    if M <= 1:
        return None

    # 在 [lambda0 - Δλ/2, lambda0 + Δλ/2] 范围内均匀采样 M 个波长
    lambdas = np.linspace(lambda0_nm - delta_lambda / 2, lambda0_nm + delta_lambda / 2, M)

    # 高斯光谱权重
    # FWHM = 2.355 * sigma，所以 sigma = FWHM / 2.355
    sigma = delta_lambda / 2.355
    weights = np.exp(-((lambdas - lambda0_nm) ** 2) / (2 * sigma ** 2))
    weights /= weights.sum()  # 归一化权重

    # 对每个波长独立传播，然后加权求和强度
    total_intensity = np.zeros((Nx, Ny), dtype=np.float64)

    for lam, w in zip(lambdas, weights):
        U = propagate(U0_template.copy(), Nx, Ny, dx, dy, lam, dz, padding, band_limited)
        I = (np.abs(U) ** 2).astype(np.float64)
        total_intensity += w * I

    return total_intensity


def spatial_coherence(
    U0_template: np.ndarray,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    lambda_nm: float,
    K: int,
    dz: float,
    padding: bool,
    band_limited: bool,
) -> np.ndarray:
    """空间相干性：一维扩展光源的非相干求和。

    物理模型：光源不是点光源，而是沿 x 方向有一定尺寸的扩展光源。
    可以看作是 K 个不同角度的平面波从不同方向入射。
    每个方向独立传播，最终的强度图样是各方向强度图样的加权和：
        I_total = Σ w_k * |U(θ_k)|^2

    权重 w_k 是高斯分布，模拟均匀发光的光源。

    注意：目前只沿 x 方向（一维）做角度叠加。二维角度网格的计算量
    是 K^2，对于实时交互来说太大，所以限制为一维。

    参数：
        U0_template: 孔径光场模板，形状 (Nx, Ny)，complex128
        Nx, Ny: 网格尺寸
        dx, dy: 像素间距，单位 μm
        lambda_nm: 波长，单位 nm
        K: 采样方向数。通常取 3, 5, 7。
        dz: 传播距离，单位 mm
        padding: 是否零填充
        band_limited: 是否带限

    返回：
        intensity: 总强度图，形状 (Nx, Ny)，float64。
            如果 K <= 1，返回 None（点光源无需此计算）。
    """
    from lightprop.angspec import propagate

    if K <= 1:
        return None

    # 模拟扩展光源：在 [-max_theta, +max_theta] 范围内采样 K 个角度
    max_theta = 5.0  # 最大偏轴角度，单位度
    thetas = np.linspace(-max_theta, max_theta, K)

    # 高斯角度权重
    sigma = max_theta / 2.355
    weights = np.exp(-(thetas ** 2) / (2 * sigma ** 2))
    weights /= weights.sum()

    # 对每个方向独立计算：先施加偏轴相位，再传播
    total_intensity = np.zeros((Nx, Ny), dtype=np.float64)

    for theta, w in zip(thetas, weights):
        U = U0_template.copy()
        # 施加偏轴相位（只沿 x 方向）
        U = apply_tilt(U, Nx, Ny, dx, dy, lambda_nm, theta, 0.0)
        U = propagate(U, Nx, Ny, dx, dy, lambda_nm, dz, padding, band_limited)
        I = (np.abs(U) ** 2).astype(np.float64)
        total_intensity += w * I

    return total_intensity
