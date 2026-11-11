# 主线阶梯：从 Rayleigh 到被测量的天空

生产入口的薄调用，不是新求解器。每层恰好叠加一个要素，逐层递进；物理叙事的
单源是 [阶梯文档](../../docs/physics/LAYER_STORY.md)（与按组件组织的
[近似图](../../docs/physics/APPROXIMATIONS.md) 互补互链），正式图片/参数见
[生产图录](../../figures/README.md)。

```bash
# 默认：全阶梯 11 次生产 solve（10 确定性 + 1 MC），只写本主题 out/ 和 figures/（不进 Git）
uv run python -m experiments.layerstory.layer_story
# 缩格冒烟（同一 solve 清单，几何缩格）
uv run python -m experiments.layerstory.layer_story --quick
# 预注册增补（2026-11-12 已执行一次）：恰一次 L3 dome solve，写
# data/layer_story_v3.npz 与 layers_v3.{csv,md}；拒绝覆盖既有 run record
uv run python -m experiments.layerstory.layer_story --augment-l3-dome
# 显式主题测试；默认 pytest 不导入 experiments
uv run pytest -q experiments/tests
# 存档覆盖审计（只读提交 NPZ，写 out/audit_coverage.{md,json}；逐 run solve 版本）
uv run python -m experiments.layerstory.refresh_figures \
  --from-npz experiments/layerstory/data/layer_story_v3.npz --audit-only
# 九张正式图（图 1–9，零 solve）：谱 / 透视 / 等距鱼眼重投影 + 计算标注；
# 普通运行只写主题 out/figures/，--publish-figures 才写根 figures/ 的 *_v3.png
uv run python -m experiments.layerstory.refresh_figures \
  --from-npz experiments/layerstory/data/layer_story_v3.npz --figure all
```

## 固定输入与阶梯

- 三个介质，逐层只换一个要素（380–780 nm / 5 nm，US1976 空气，地面反照率 0.1，
  海平面观察者，折射全程开启，太阳输入为折射前方向、方位 0°）：
  1. **L1** `presets.pristine(WL, ozone_du=0)`——纯 Rayleigh（无 O₃/NO₂/气溶胶）；
  2. **L2** `presets.clear(WL, aod=0.05, ozone_du=0)`——叠加 Hinterland 边界层
     气溶胶 AOD550=0.05（指数廓线 H=1.5 km）；
  3. **L3 起** 冻结介质：O₃ 300 DU + NO₂ 柱量 4×10¹⁹ m⁻²（4×10¹⁵ cm⁻²，文献
     典型范围 2–6×10¹⁵ cm⁻² 的中值；`Medium` 内建廓线只提供形状、按此柱量归一）。
     后续各层只改阶选择、输运几何或叠加云层，介质不再变化。
- 云层 L6：`cloud.stratus()`（tau550=20，底 0.5 km，顶 1.5 km，水云）经
  `cloud_layer` 以 `hybrid-sos` 契约叠加；观察者低于云底。
- 穹顶网格：日景层 25×37，暮光层 11×23，云层对 11×21（引擎按视角计价、
  梯度平滑，故暮光/云取粗网格；网格写进 `config` 元数据）。原阶梯中 L3
  只取天顶谱行；2026-11-12 起预注册增补 run `l3_o3_dome_3` 把 L3 态补成
  完整 25×37 dome（恰一次 `render.solve_dome`，见下文与
  `data/l3_dome_preregister.json`）。
- MC 请求 `quality=low`、单 seed=7、网格 3×4、5 锚、每锚 200000 光子——
  **qualitative / budget-limited**，不做多 seed 变异性研究（偏离 compare.py 之处，
  见下）。实际 clamp/舍入值只读结果 `effective`。

## 对照协议（每层差异的归因基线）

| 层 | 叠加要素 | 差异基线（对照态） |
|---|---|---|
| L1 | 纯 Rayleigh | 无（叙事起点） |
| L2 | +Mie 气溶胶 | L1 态 @30°（复用 L1 solve） |
| L3 | +O₃/NO₂ | L2 介质 @3° 重解（桥接态） |
| L4 | +多次散射 | L3 介质 @30° single-only 重解（桥接态） |
| L5 | +球面几何与折射 | −4° 场景内：`spherical-single` 基线 vs `spherical-mc` vs `plane-sos`（平面世界对照） |
| L6 | +云 | 同场景 30° 契约匹配对：`hybrid-sos` clear vs +stratus |
| L7 | 与观测对表 | 无新计算；引用既有 validation 证据，见阶梯文档 |

每层差异 = 同场景、单变量（换介质/加阶次/换输运几何/加云）。**跨要素加和不
成立也不提供**：多次散射对光学厚度组成非线性，把「L1+L2 贡献」相加没有物理
意义——closure 检查只在单次 solve 内部（见下）。

## 文件与读回

[layer_story.npz](data/layer_story.npz) 保存每次调用的 total 与分量数组、
波长、实际方向、未 tone-map 的 XYZ/xy/Lv、墙钟、options/effective/diagnostics/
approximations 与摘要；overlay 另存衰减前 single/multiple 与地面绑定材料。
[layer_story_v3.npz](data/layer_story_v3.npz)（2026-11-12 增补档）在其上
逐位保留原 11 run 全部字段，追加增补 run `l3_o3_dome_3` 与**逐 run
`solve_version`**（原 11 run 显式映射原全局版本；增补 run 记录自己的
compute-time HEAD 与工作区指纹）、来源档案 hash 与混合来源说明——它是
混合来源档案，不是"同一版本 12 次求解"。原档与
`layers.csv`/`layers.md` 原字节保留；v3 调用表是
[layers_v3.csv](data/layers_v3.csv) / [layers_v3.md](data/layers_v3.md)，
运行凭据在 `data/l3_dome_run.json`（预注册
`data/l3_dome_preregister.json` 先于 solve 落盘）。
分量为 W m⁻² sr⁻¹ nm⁻¹，Lv=683Y，单位 cd m⁻²。字符串是普通文本，
`np.load(..., allow_pickle=False)` 即可读。CSV 表列实际方向
和分量，[调用记录](data/layers_v3.md) 列采用版本、全部有效设置、诊断与各次耗时。

```python
import numpy as np
from skycolor import color
with np.load('experiments/layerstory/data/layer_story.npz', allow_pickle=False) as d:
    rad = d['l3_o3_3/total']
    xyz = color.spectrum_to_xyz(rad, d['wl'])
```

同路径分量逐位闭合（clear 族 `total = single + multiple`；overlay 族
`total = clear + cloud`，其中 `clear` 是**透射衰减后**的清空场）。三组分解恒等式
逐位断言：L4 桥接态 == 全阶 solve 的 single 分量；L5 基线 == MC solve 的 single
分量；overlay 的衰减前清空场 == 契约匹配的 clear solve。**不加跨 solve 的
闭合**（见对照协议）。

## 共同展示与定标记录

所有图最终调用同一个 `render_result`，分三组锁定参数（组内无逐图自动曝光/
白平衡；`adaptation=None` 绝对色度，`sun_disk=False`，ACES，
`preserve_chromaticity=True`）：

| 组 | 层 | exposure | 定标记录 |
|---|---|---|---|
| day | L1/L2/L4 | 0.03 | 复用 approximations 主题已定标值（同一 tonemap 基准） |
| twilight | L5 三态 | 10.0 | 由首个真实 −4° 结果一次性定标锁定（天顶 Lv≈4.4 cd m⁻²、光带 4–23 cd m⁻²；红橙光带可读且为更亮的 MC 面板留动态余量） |
| cloud | L6 对 | 0.03 | 云下天顶 Lv≈6.4×10³ cd m⁻² 在该曝光下中灰；与日景组同值 |

日景组 `ground=False`（纯天空）；云组 `ground=True`（阴天地板是 L6 叙事的
一部分，地面照明由 solve 绑定材料给出；下地平像素按生产显示行为处理）。
暮光组 `ground=False`。不画太阳标记；只有天空漫射光。

## 表达力核查表（Step 0 数值探针结论）

| 层 | 载体入口 | 结论 | 证据 |
|---|---|---|---|
| L1 | solve_dome + plane-sos single-only | 可表达 | 天顶 λ⁻⁴ 斜率 −3.74（420–680 nm 拟合） |
| L2 | 同上 | 可表达 | AOD 0.05 可见性充分：地平 Lv ×2.6、45° 向日 ×3.8（Step 0 探针 46×91 默认网格读数；生产 25×37 网格为 ×2.59/×3.92——探针网格在地平附近无 86.25° 节点，两者非严格同一观察量）；不需「加浊变体」 |
| L3 | rt.solver.solve 天顶行 | 可表达 | Chappuis 带：560–610 nm 比值 0.674（相对无 O₃ 桥接态）；NO₂ 天顶最大相对变化 2.6%（可见、次要，如实声明） |
| L4 | solve_dome + plane-sos 双阶 | 可表达 | 反太阳侧多次散射占比 24–37%（550 nm） |
| L5 | solve_dome + spherical-single / spherical-mc | 可表达 | −4° 实跑通过；single 天顶 Lv 4.41、MC 6.08 cd m⁻²（+38% 填充） |
| L5 对照 | solve_dome + plane-sos @−4° | 可表达（采纳） | plane 族**不拒绝**负太阳角且物理可判读：连续柱透射让高层在地平下仍被「照亮」。天顶量级与球面几乎相同（550 nm 5.887×10⁻⁵ vs 5.870×10⁻⁵，×1.003——单点量级可「碰对」），但角向结构全错：严格地平（90° 天顶节点）平面世界精确为零，而球面 single 同方向保留 2.70×10⁻⁵（向日侧）——没有地影、没有暮光弧 |
| L6 | solve_dome + hybrid-sos ± cloud_layer | 可表达 | 契约匹配对跑通；云下天顶 ×4.7、谱平坦度 0.068（近白）；观察者低于云底成立 |
| L7 | — | 无新计算 | 引用既有 validation 证据（阶梯文档） |

介质步进核查（对象回读断言，`compute()` 每次运行执行）：L3 = L2 + {O₃, NO₂}
逐项成立——albedo 数组、观察者高度、折射开关、地面气压相等；两者复用同一个
共享 Hinterland 气溶胶对象（AOD550=0.05，廓线与水平场参数逐项相等）；仅气体
柱量不同（0 → 300 DU，0 → 4×10¹⁹ m⁻²）。「每层只叠加一个要素」由机制证据
而非构造调用信任支撑。

架构反馈：**无**——七层全部经现有 `Medium`/`Scene`/四近似契约干净表达，
`skycolor/` 零 diff。

NO₂ 取值决策：`Medium._NO2_PROFILE` 内建廓线的自然柱量约 3.0×10¹⁶ cm⁻²，比
文献典型平流层柱量（2–6×10¹⁵ cm⁻²）高约一个量级，不作为取值；按文献范围取
中值 4×10¹⁹ m⁻²，廓线形状仍用内建表（`no2_column` 参数按柱量归一）。

## 存档覆盖审计与逐位置专属图（v3）

`refresh_figures.py` 从提交的增补档 `data/layer_story_v3.npz` **零 solve**
重画两篇文档的九张逐层配图（图 1 总览 + 图 2–9 逐位置）：总览为 **2×4
八张渲染、无曲线面板**（曲线由逐层谱图承载），每层专属谱图、透视渲染，
以及天顶居中的 180° 等距鱼眼（`fisheye_grid` 等距分支 + `sample_dome` +
`color.render_rgb`，同一场的显示重采样，不增加物理信息）。
地平线/方位刻度/太阳方向标注全部由同一相机网格计算，太阳只标**折射前
输入方向**（`sun_disk=False` 不变，地平下太阳只给地平边界指示）。
2026-11-12（task-figures-polish）几何升级：鱼眼统一**方形轴盒**（等比例、
标注后图像边界对称、无像素刻度）；透视面板改用存档相机参数经 pinhole
映射（`f=(w/2)/tan(hfov/2)`、半像素中心）诚实导出的**相机相对角 α/β
整数刻度**（刻度不等距是 pinhole 非线性，不是对线性像素硬换标签；
pitch/yaw 仍驱动生产方向场，地平/太阳标注保持绝对方向）；共享谱 helper
补 `Wavelength [nm]` 横轴。图题脚注**按 run 生成**（网格/间距来自档案；
截断零解释只给实际含零的 run，并报方位区间与透视窗内外判定）。

**⑦ 实测结论（owner 探针，2026-11-12）**：`l5_single_-4` 的地平截断零仅在
反日侧 rel-az 147.3–180°（5/23 格），位于 yaw 0° 透视帧 ±50° 日侧方位窗
之外——透视视图是"看不到"该方位区，不是"压缩掉"了结构；日侧窗内两相机
均有采样且方位分辨率相当（透视 0.156°/px 对鱼眼缘 0.179°/px）。
`l5_plane_-4` 整行地平零是平面世界构造（无地平辐射），不是截断产物。

`--audit-only` 先做覆盖审计：逐 run 网格端点/行数/地面材料/太阳元数据/
**逐 run solve 版本** + 逐图可行性（缺 run 的图按名报告，不抛 KeyError、
不自动 solve），写 `out/audit_coverage.{md,json}`；图定义（ID/曲线方向/
视图/消费者）单源在入口的 `FIGURES` 注册表，逐图命令与参数导出
`out/refresh_manifest.json`（含逐 run 求解来源）。普通运行只写主题
`out/`；`--publish-figures` 才写根 `figures/` 的 `*_v3.png` 命名图。

增补协议（`--augment-l3-dome`，2026-11-12 执行一次）：tracked 预注册
（源档 hash、唯一请求、300s 预算、停止条件、天顶容差 rtol=1e-10）先于
solve 落盘并提交；入口校验预注册后**恰派发一次** `render.solve_dome`
（10.29s，实测），天顶 37 方位对旧 `l3_o3_3` 谱交叉核验通过（最大偏差
1.7e-18），旧 11 run 字段逐位不动、原档 hash 不变；超时/失败/核验不成立
只写 failure record 并停止，绝不第二次求解兜底。

覆盖审计结论（2026-11-12）：原阶梯 L1/L2/L4 穹顶 25×37、L5 11×23、
L6 11×21、L3 两态 1×1 天顶行（只出谱图）；增补 run `l3_o3_dome_3`
25×37 完整 dome（`solve_version=3b9d19c`，clean-worktree）。曲线方向
86.25°/45°/81° 等均为精确网格节点。

v1 两张网格图与 v2 八张逐位置图的字节快照（tracked，含逐图 hash）在
[historical/](historical/README.md)；根 `figures/` 当前集自 2026-11-12 起
为九张 `layer_story_*_v3.png`。

## 与 approximations 主题的偏离

- **MC 单 seed、定性**：本主题的 MC 只演示「高阶散射填充暮光」这一个定性
  对比，不做三 seed 变异性研究（那是 compare.py 的职责）。
- **closure 口径更窄**：单 solve 内部分量加和 + 三组桥接恒等式；不提供跨
  介质/跨要素加和（物理上不成立）。
- **逐层网格不同**：日景 25×37、暮光 11×23、云 11×21；原阶梯 L3 只取
  天顶谱行（v3 增补后另有完整 dome run，见上）。网格是叙事载体参数，
  不是精度声明。
- **云组 ground=True**：阴天地板进入叙事；其余组纯天空。
