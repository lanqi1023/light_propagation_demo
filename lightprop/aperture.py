"""光阑生成模块。

本模块负责在孔径平面上生成入射光场 U0。所有数组使用形状 (Nx, Ny) 的 complex128，
符合 numpy 默认的 (row, col) = (y, x) 约定。

坐标系说明：
    - Nx：行数，对应 y 方向
    - Ny：列数，对应 x 方向
    - xs：x 坐标数组，形状 (Ny,) 或 (Nx, Ny)
    - ys：y 坐标数组，形状 (Nx,) 或 (Nx, Ny)
    原点 (0, 0) 位于网格中心。
"""
import numpy as np


def _grids(Nx: int, Ny: int, dx: float, dy: float):
    """生成以原点为中心的 1D 坐标数组。

    参数：
        Nx: 行数（y 方向）
        Ny: 列数（x 方向）
        dx: x 方向像素间距，单位 μm
        dy: y 方向像素间距，单位 μm

    返回：
        xs: x 坐标数组，形状 (Ny,)，范围 [-xc*dx, xc*dx]
        ys: y 坐标数组，形状 (Nx,)，范围 [-yc*dy, yc*dy]
    """
    xc = (Ny - 1) / 2.0
    yc = (Nx - 1) / 2.0
    xs = (np.arange(Ny) - xc) * dx   # x 随列索引变化
    ys = (np.arange(Nx) - yc) * dy   # y 随行索引变化
    return xs, ys


def generate(
    aperture_type: str,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    params: dict,
    w0: float = 100.0,
) -> np.ndarray:
    """生成入射光场 U0。

    透射区振幅为 1，阻挡区为 0。可选施加高斯包络 exp(-(x^2+y^2)/w0^2)。

    参数：
        aperture_type: 光阑类型，见 Params.aperture_type
        Nx, Ny: 网格尺寸
        dx, dy: 像素间距，单位 μm
        params: 光阑参数字典
        w0: 高斯束腰半径，单位 mm。
            当 0 < w0 < 100 时施加高斯包络；
            当 w0 >= 100 时视为平面波，跳过包络。

    返回：
        U0: 形状 (Nx, Ny) 的 complex128 数组，表示孔径平面的光场。
    """
    # 使用 meshgrid 生成二维坐标网格
    # indexing="xy" 表示第一个维度对应 y，第二个维度对应 x
    # 因此 xs 和 ys 的形状都是 (Nx, Ny)
    xs, ys = np.meshgrid(*_grids(Nx, Ny, dx, dy), indexing="xy")
    # xs shape (Nx, Ny), ys shape (Nx, Ny)

    # 初始化光场为 0
    U0 = np.zeros((Nx, Ny), dtype=np.complex128)

    # 根据光阑类型生成不同的孔径分布
    if aperture_type == "slit":
        # 单缝：沿 x 方向的无限长狭缝，宽度为 params.width
        half_w = (params.get("width", 100.0) / 2.0)
        U0[np.abs(xs) <= half_w] = 1.0

    elif aperture_type == "doubleSlit":
        # 双缝：两个平行狭缝，宽度为 params.width，间距为 params.separation
        half_w = (params.get("width", 50.0) / 2.0)
        half_s = (params.get("separation", 200.0) / 2.0)
        mask = (np.abs(xs - half_s) <= half_w) | (np.abs(xs + half_s) <= half_w)
        U0[mask] = 1.0

    elif aperture_type == "circle":
        # 圆孔：半径为 params.radius 的圆形孔径
        r2 = params.get("radius", 100.0) ** 2
        mask = (xs ** 2 + ys ** 2) <= r2
        U0[mask] = 1.0

    elif aperture_type == "rectangle":
        # 矩形孔：宽 params.width，高 params.height
        half_w = (params.get("width", 150.0) / 2.0)
        half_h = (params.get("height", 150.0) / 2.0)
        mask = (np.abs(xs) <= half_w) & (np.abs(ys) <= half_h)
        U0[mask] = 1.0

    elif aperture_type == "free":
        # 自由空间：无光阑，全场透射
        U0[:] = 1.0

    elif aperture_type == "upload":
        # 上传图片：由前端预处理好像素后直接设置，此处跳过
        pass

    else:
        # 默认退化为单缝
        half_w = (params.get("width", 100.0) / 2.0)
        U0[np.abs(xs) <= half_w] = 1.0

    # 施加高斯包络（如果 w0 < 100 mm）
    if 0 < w0 < 100:
        # 注意：w0 单位是 mm，需要转换为 μm 与 dx, dy 单位统一
        w0_sq = (w0 * 1e3) ** 2   # mm^2 -> μm^2
        g = np.exp(-(xs ** 2 + ys ** 2) / w0_sq)
        U0 *= g

    return U0
