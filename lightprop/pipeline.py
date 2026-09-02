"""计算管线编排器。

本模块是 lightprop 的核心入口，负责将各个物理模块串联成完整的传播计算流程：
    1. 生成孔径光场（aperture.py）
    2. 施加偏轴相位（optics.py，可选）
    3. 角谱法传播（angspec.py）
    4. 后处理：强度、相位、截面提取

设计原则：
    - compute() 是唯一公开入口，外部不应直接调用子模块
    - 所有单位转换都在子模块内部完成，pipeline 只传递原始参数
    - 支持自动网格升级：当带限滤波导致结果退化时，自动使用更大网格重算
"""
import time

import numpy as np

from lightprop.types import Params, Result
from lightprop.aperture import generate as generate_aperture
from lightprop.optics import apply_tilt, temporal_coherence, spatial_coherence
from lightprop.angspec import propagate as angspec_propagate


def _auto_upgrade_grid(params: Params, intensity: np.ndarray) -> tuple[Params, bool]:
    """如果带限滤波导致结果退化，自动升级网格尺寸。

    判定条件：结果中_unique 强度值数量 <= 4。
    这意味着带限滤波过度，几乎所有频率都被滤除，输出几乎均匀。

    升级策略：从小到大尝试 [512, 256, 128]，保持物理窗口 Lx, Ly 不变，
    即保持 dx, dy 不变，只增大 Nx, Ny。

    参数：
        params: 原始参数
        intensity: 传播后的强度数组，用于判断是否退化

    返回：
        (new_params, upgraded): 新参数和是否升级的标志
    """
    if params.band_limited and intensity is not None:
        unique_vals = len(np.unique(intensity))
        if unique_vals <= 4:
            # 结果退化，尝试升级网格
            for new_n in [512, 256, 128]:
                if params.Nx < new_n and params.Ny < new_n:
                    upgraded = Params(**{**params.__dict__})
                    upgraded.Nx = new_n
                    upgraded.Ny = new_n
                    # 保持物理窗口 Lx, Ly 不变：dx = Lx / Nx
                    upgraded.dx = (new_n * params.dx) / upgraded.Nx
                    upgraded.dy = (new_n * params.dy) / upgraded.Ny
                    return upgraded, True
    return params, False


def compute(params: Params) -> Result:
    """执行完整的传播计算。

    这是用户应该调用的唯一入口函数。

    参数：
        params: 计算参数，参见 Params 类说明

    返回：
        Result: 包含强度图、相位图、截面图和元数据的结果对象

    计算流程：
        1. 根据参数生成孔径光场 U0
        2. 如果启用偏轴照明，施加线性相位调制
        3. 根据非理想效应模式选择传播策略：
           - 空间相干性 only：多角度叠加
           - 时间相干性 only：多波长叠加
           - 两者都开启：M×K 双重循环叠加
           - 标准模式：单次角谱法传播
        4. 检查结果是否退化，必要时自动升级网格重算
        5. 归一化强度、相位，提取中心截面
        6. 打包返回 Result
    """
    t0 = time.perf_counter()

    Nx, Ny = params.Nx, params.Ny
    dx, dy = params.dx, params.dy
    lambda_nm = params.lambda_nm
    dz = params.dz

    # ========== Step 1: 生成孔径光场 ==========
    # 根据光阑类型和参数，生成 z=0 平面的初始光场
    U0 = generate_aperture(
        params.aperture_type,
        Nx, Ny, dx, dy,
        params.aperture_params,
        params.w0,
    )

    # ========== Step 2: 施加偏轴相位 ==========
    # 如果启用偏轴照明，在孔径平面引入线性相位因子
    if params.tilt_on:
        U0 = apply_tilt(U0, Nx, Ny, dx, dy, lambda_nm,
                        params.tilt_x_deg, params.tilt_y_deg)

    # ========== Step 3: 传播计算 ==========
    # 根据非理想效应模式选择不同的传播策略
    intensity = None
    phase = None

    if params.spatial_on and not params.temporal_on:
        # 空间相干性 only：K 个方向叠加
        intensity = spatial_coherence(
            U0, Nx, Ny, dx, dy, lambda_nm,
            params.K, dz, params.padding, params.band_limited,
        )
        if intensity is None:
            # K=1 时退化为标准传播
            Uz = angspec_propagate(U0, Nx, Ny, dx, dy, lambda_nm, dz,
                                   params.padding, params.band_limited)
            intensity = np.abs(Uz) ** 2
        phase = np.angle(U0)

    elif params.temporal_on and not params.spatial_on:
        # 时间相干性 only：M 个波长叠加
        intensity = temporal_coherence(
            U0, Nx, Ny, dx, dy, lambda_nm,
            params.delta_lambda, params.M, dz,
            params.padding, params.band_limited,
        )
        if intensity is None:
            # M=1 时退化为标准传播
            Uz = angspec_propagate(U0, Nx, Ny, dx, dy, lambda_nm, dz,
                                   params.padding, params.band_limited)
            intensity = np.abs(Uz) ** 2
            phase = np.angle(Uz)
        else:
            # 时间相干性下相位无物理意义，使用输入相位代替
            phase = np.angle(U0)

    elif params.temporal_on and params.spatial_on:
        # 组合模式：M×K 双重循环
        # 计算量 = M * K 次传播，需要特别注意性能
        from lightprop.angspec import propagate as _prop

        # 生成 M 个波长及其权重
        lambdas = np.linspace(
            lambda_nm - params.delta_lambda / 2,
            lambda_nm + params.delta_lambda / 2,
            params.M,
        )
        sigma = params.delta_lambda / 2.355
        w_lam = np.exp(-((lambdas - lambda_nm) ** 2) / (2 * sigma ** 2))
        w_lam /= w_lam.sum()

        # 生成 K 个角度及其权重
        max_theta = 5.0
        thetas = np.linspace(-max_theta, max_theta, params.K)
        sigma_t = max_theta / 2.355
        w_theta = np.exp(-(thetas ** 2) / (2 * sigma_t ** 2))
        w_theta /= w_theta.sum()

        total = np.zeros((Nx, Ny), dtype=np.float64)
        final_U = None

        for lam, wl in zip(lambdas, w_lam):
            for theta, wt in zip(thetas, w_theta):
                U = U0.copy()
                U = apply_tilt(U, Nx, Ny, dx, dy, lam, theta, 0.0)
                Uz = _prop(U, Nx, Ny, dx, dy, lam, dz,
                           params.padding, params.band_limited)
                I = np.abs(Uz) ** 2
                total += wl * wt * I
                if final_U is None:
                    final_U = Uz

        intensity = total
        phase = np.angle(final_U) if final_U is not None else np.angle(U0)

    else:
        # 标准单波长传播
        Uz = angspec_propagate(U0, Nx, Ny, dx, dy, lambda_nm, dz,
                               params.padding, params.band_limited)
        intensity = np.abs(Uz) ** 2
        phase = np.angle(Uz)

    # ========== Step 3b: 自动网格升级 ==========
    # 如果带限滤波导致结果退化（<=4 个唯一值），自动升级网格重算
    params, upgraded = _auto_upgrade_grid(params, intensity)
    if upgraded:
        Nx, Ny = params.Nx, params.Ny
        dx, dy = params.dx, params.dy
        U0 = generate_aperture(
            params.aperture_type,
            Nx, Ny, dx, dy,
            params.aperture_params,
            params.w0,
        )
        if params.tilt_on:
            U0 = apply_tilt(U0, Nx, Ny, dx, dy, lambda_nm,
                            params.tilt_x_deg, params.tilt_y_deg)
        Uz = angspec_propagate(U0, Nx, Ny, dx, dy, lambda_nm, dz,
                               params.padding, params.band_limited)
        intensity = np.abs(Uz) ** 2
        phase = np.angle(Uz)

    # ========== Step 4: 后处理 ==========
    # 强度归一化到 uint8 [0, 255]
    max_int = intensity.max()
    if max_int > 0:
        intensity_uint8 = np.round(255.0 * np.sqrt(intensity / max_int)).astype(np.uint8)
    else:
        intensity_uint8 = np.zeros((Nx, Ny), dtype=np.uint8)

    # 相位映射到 uint8 [0, 255]
    phase_uint8 = np.round(((phase / np.pi + 1.0) / 2.0) * 255).astype(np.uint8)
    phase_uint8 = np.clip(phase_uint8, 0, 255)

    # 提取中心截面
    cy = Ny // 2
    cx = Nx // 2
    cross_x = intensity[:, cy].astype(np.float64)   # 沿 x 方向（列），长度 Ny
    cross_y = intensity[cx, :].astype(np.float64)   # 沿 y 方向（行），长度 Nx

    # 计时
    t1 = time.perf_counter()

    # 计算元数据
    Lx = Nx * dx / 1000.0   # mm
    Ly = Ny * dy / 1000.0   # mm
    fresnel_num = (Lx * Lx) / (4.0 * lambda_nm * 1e-6 * dz)
    sampling_ok = params.band_limited or (dx <= params.lambda_nm / 2000.0)

    info = {
        "fresnelNum": float(fresnel_num),
        "Lx": float(Lx),
        "Ly": float(Ly),
        "nx": Nx,
        "ny": Ny,
        "computationTime": int((t1 - t0) * 1000),
        "samplingOk": bool(sampling_ok),
        "maxInt": float(max_int),
    }

    return Result(
        intensity=intensity_uint8,
        phase=phase_uint8,
        cross_x=cross_x,
        cross_y=cross_y,
        info=info,
    )
