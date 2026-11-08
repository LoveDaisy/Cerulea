# experiments/ — 现象研究主题

每个子目录是一个研究主题：脚本是对生产入口（`skycolor.presets` / `render.sky_dome` /
`render.render_scene` / `solve` / `solve_dome`）的**薄调用**，输出（`figures/`、`out/`）
归主题目录自身、默认不进 git；被选定的生产代表图才显式写入根
`figures/`，见[当前图录](../figures/README.md)。

先看[近似主题](approximations/README.md)：固定两场景、同一入口、可回读谱与诊断、
共曝光图，避免先从历史长扫描推断当前能力。

## 运行入口

```bash
uv run python -m experiments.approximations.compare
uv run python -m experiments.day.day_cycle [--quick]
uv run python -m experiments.twilight.twilight [--quick]
uv run python -m experiments.twilight.afterglow
uv run python -m experiments.aerosol.aerosol
uv run python -m experiments.heterogeneity.heterogeneity
uv run python -m experiments.altitude.altitude_latitude
uv run python -m experiments.bluesun.blue_sun
uv run python -m experiments.clouds.clouds
```

`--quick`（day/twilight）：几何缩格的有界冒烟（光谱网格不变——色度辅助函数在其上积分）。
入口约定：从仓库根以 `-m` 运行；主题输出写各自目录，`save()` 接收显式目的路径，
import 无目录创建副作用。

## 主题

| 目录 | 内容 | 对应现象 |
|---|---|---|
| [approximations/](approximations/README.md) | 固定30°/3°，三确定性路线与有限MC | 加阶次与换传播/算法分开；正式生产代表图 |
| `day/` | 日变化色彩轨迹与全景条带 | 太阳 -18°…+70°扫描；深点仍用默认SOS，不认证深暮光 |
| `twilight/` | 暮光亮度/色度扫描（Patat 2006 对照）与鱼眼序列、晚霞/火山暮光 | 暮光渐进与 afterglow |
| `aerosol/` | 气溶胶负荷/类型/霾渐变 | AOD 与成分对天空色彩的影响 |
| `heterogeneity/` | 非均匀气溶胶场 | 水平非均匀性 |
| `altitude/` | 海拔/纬度 | 观察者高度与纬度响应 |
| `bluesun/` | 蓝太阳 | 大颗粒气溶胶下的太阳色 |
| `clouds/` | 云（清/阴、鱼眼、日落序列、反照率） | 云层对天空与地面照明 |
| `cssp/` | CSSP 观测输入工具与冻结包（hermetic） | 真实观测对比的输入端，见其 README |
| `retired/` | 退休主题索引与历史图（代码在固定历史提交） | 复跑入口与证据定位，见其 README |

## 共用辅助

`common.py`：`save(fig, path)`（显式路径、写入时建目录）、`strip`/`row`（图像拼接）、
`chromaticity_of`（81 波长工作网格上的 CIE 转换）、`panel_label`。

## 边界

- 实验**单向**使用生产 API；生产不导入本目录。局部纯解析/光学探针不强塞完整天空求解器。
- 主题 `figures/`/`out/` 的新运行产物默认不入库；`retired/figures/`及主题中显式
  跟踪的`historical_*.png`保全迁入旧图，不被普通运行覆盖。旧图不代表当前版本：
  aerosol 3张（自动适应）、heterogeneity 2张（仅一次项水平场）、bluesun 2张
  （光学探针，旧生成版本未知）、clouds 3张（pre-M10）。去向见根图录。
  `approximations/data/`仅选入本轮实际采用的谱/表/诊断。
- 数值脚本的历史结论见 `retired/README.md`；硬编码旧数值的脚本不充当当前模型输出。
