# 近似比较：两个固定场景

这是生产入口的薄调用，不是新求解器或精度排名。物理含义见
[近似图](../../docs/physics/APPROXIMATIONS.md)，正式图片/参数见
[生产图录](../../figures/README.md)。

```bash
# 默认：只写本主题 out/ 和 figures/（不进 Git）
uv run python -m experiments.approximations.compare
# 正式发布可复用本次求得的普通 NPZ，不再求解
uv run python -m experiments.approximations.compare \
  --from-npz experiments/approximations/out/representative.npz --publish-figures
# 也可直接从提交的实测谱重画正式图，不产生新的物理结果
uv run python -m experiments.approximations.compare \
  --from-npz experiments/approximations/data/representative.npz --publish-figures
# 显式主题测试；默认 pytest 不导入 experiments
uv run pytest -q experiments/approximations/tests experiments/tests
```

## 固定输入与顺序

- `presets.clear(WL_COARSE, aod=0.05, ozone_du=300)`，380–780 nm / 5 nm，
  `HinterlandAerosol`、指数气溶胶廓线 H=1.5 km；默认 US1976 空气/臭氧廓线，
  NO₂ 柱=0，地面反照率=0.1。均匀、无云、观察者=不透明底部=海平面。
- 折射开启；太阳输入是**折射前**高度 30°/3°、方位 0°。源为当前生产
  `solar_source.py`，不是退休研究实现。观察/漫射路径不是全阶折射。
- 每场景同一 25×37 半穹顶（天顶0…90°、相对方位0…180°）。表取天顶、
  45°两侧与近地平两侧；近地平实际节点为86.25°（高度3.75°），不是88°。
- 先 `plane-sos`，再 `hybrid-sos`，再真实调用 `spherical-single`；30°场景
  再调用 `spherical-mc` 固定种子7/17/27，然后3°三确定性调用。第一次含冷启动，
  后续共享暖光学；耗时不是公平性能排名。SOS 保持默认32矩/32 μ/12 Fourier、
  max_orders=40、tol=1e-3、delta-M开启；一次项 n_gauss=3、chunk=512。
- MC请求 `quality=low`，网格3×4、5锚、每锚200000光子；实际clamp/舍入值
  只读结果 `effective`，详见[调用记录](data/representative.md)。没有深点长跑。

## 文件与读回

[representative.npz](data/representative.npz) 保存每次调用的 total/single/multiple、
波长、实际方向、未tone-map的XYZ/xy/Lv、墙钟、options/effective/diagnostics/
capabilities与摘要；MC另存stderr。分量为 W m⁻² sr⁻¹ nm⁻¹，Lv=683Y，单位cd m⁻²。
字符串是普通文本，`np.load(..., allow_pickle=False)`即可读，不依赖对象序列化。
[CSV表](data/representative.csv)列实际方向和分量，
[调用记录](data/representative.md)列采用版本、全部有效设置、诊断与各次耗时。

```python
import numpy as np
from skycolor import color
with np.load('experiments/approximations/data/representative.npz', allow_pickle=False) as d:
    rad = d['e3_hybrid-sos/total']
    xyz = color.spectrum_to_xyz(rad, d['wl'])
    Lv = 683 * xyz[:, 1]
    xy = color.chromaticity(xyz)
```

同路径分量逐位闭合；球面single逐位等于hybrid single；plane/hybrid高阶分量逐位
相同。后一个条件允许定位总差异来自两种一次项，但仍不能称为纯曲率效应：平面
一次项带SOS的矩表示/delta-M/离散，球面一次项直接用相函数积分。

## 共同展示与MC边界

所有近似/场景图最终调用同一个 `render_result`：perspective 640×320、hfov100°、
pitch14°、yaw0°，exposure=0.03、ACES、adaptation=None、preserve_chromaticity=True、
sun_disk=False、ground=False。曝光始终固定，无逐图自动适应。下地平像素按生产
显示行为处理，不表示地面物理；不画太阳标记。只有天空漫射光，不是太阳盘成像。

MC仅 **qualitative / budget-limited**：三seed结果保留，不挑seed，不合成置信区间；
stderr包含角向平滑偏差floor和光谱插值传播，不是总物理误差；跨波长相关性未知，
不把stderr相加当作Lv置信限。引擎不提供路径贡献集中度，本示例不修改MC补诊断。
未检出不等于物理高阶为零。正式深暮光负面结论仍是
[task-7档案](../../docs/twilight/DEEP_TWILIGHT_REFERENCE.md)的budget-limited-unresolved。
