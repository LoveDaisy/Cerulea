# AGENTS.md

## 项目概述

物理基础的光谱天空颜色模拟器。从太阳光谱、Rayleigh/Mie 散射、臭氧吸收和球面辐射传输
第一性原理计算天空光谱辐亮度（W m⁻² sr⁻¹ nm⁻¹），再经 CIE 色度学 → 色适应 →
tone mapping 得到颜色与图像。Python 3.12，numpy / scipy，环境由 uv + `uv.lock` 固定。

## 常用命令

### 环境

```bash
uv sync        # 从 uv.lock 重建 .venv；不要 pip install 进环境
```

### 测试

```bash
uv run pytest -q                      # 全量 252 tests，约 5 min
uv run python -m validation.run_all   # 六项物理校验，10+ min
```

### 运行

```bash
# 渲染一张图
uv run python -m skycolor.cli render --sun-elevation 3 --camera perspective \
    --size 1200x520 --pitch 14 --hfov 100 --preset clear --aod 0.05 \
    --yaw 180 --out sunset.png

# 生成实验图（figures/，脚本从仓库根目录运行）
uv run python experiments/01_day_cycle.py
```

### 代码格式化

项目没有 linter / formatter 配置；沿用现有风格（见文末“代码风格”）。

## 架构与设计

```text
.
├── skycolor/            # 库
│   ├── spectra/         # 太阳光谱、CIE CMF、O3/NO2 截面、Rayleigh
│   ├── atmosphere.py    # US1976 + 热层尾、臭氧、气溶胶层、水平场
│   ├── mie.py           # Mie 系数、解析 Legendre 矩、自适应相函数网格
│   ├── aerosol.py       # 8 成员气溶胶库；delta-M；逐波长 LRU 缓存
│   ├── geometry.py      # 太阳位置、球壳光线几何、双分支太阳弦
│   ├── refraction.py    # 折射光线追踪、Bouguer 不变量、路径长度
│   ├── rt/single.py     # 球面几何精确一次散射
│   ├── rt/multiple.py   # 逐次散射（Fourier + Gauss-Legendre）
│   ├── rt/mc.py         # 独立 Monte Carlo 参考
│   ├── rt/twilight.py   # 深暮光球面引擎（split estimator）
│   ├── rt/cloud.py      # delta-Eddington 云层
│   ├── cloud.py         # 云光学（Mie 水滴、delta-M、presets）
│   ├── color.py         # 光谱 → XYZ → sRGB、色适应、tone mapping
│   ├── render.py        # sky dome、全景/鱼眼/透视相机
│   ├── presets.py       # 命名场景
│   └── data/            # 提交在仓库里的数据表（离线可运行）
├── validation/          # check1..6 + common.py + 各轮 addendum
├── experiments/         # 生成 figures/ 的脚本（01..12）
├── tests/               # 210 tests
├── docs/                # STATUS / ROADMAP / REPORT / 专题文档 / 中文摘要
├── tools/               # build_data.py、twilight_benchmark.py
├── figures/             # 提交在仓库里的结果图
└── scratchpad/          # 本地任务管理（gitignore，不进 git）
```

### 核心模块说明

- `atmosphere.py`：`Atmosphere` 是 facade：物理介质在 frozen `Medium`（`skycolor/medium.py`），
  观察场景在 frozen `Scene`（`skycolor/scene.py`）。`at_altitude(z0)` 已改为一致截断
  （`z_bounds` 自 z0 起、地面=z0、不重标定任何柱量，Step 3c）——single/SOS/MC 首次对
  “地面在哪”同源；高度响应仍反号属已知物理缺口，不是构造问题。
- `geometry.py` / `refraction.py`：太阳弦长是暮光的关键。M10 修复后，太阳在局部水平
  以下时走“远侧 perigee→space + 近侧 perigee→point”（`up + 2*down`）。不要重做。
- `rt/single.py`：一次散射从观察者半径 `r0` 开始积分；折射开启时用
  `RefractionGeometry.apparent_elevation`。
- `rt/multiple.py`：平面平行漫射场，地面固定在 `z_bounds[0]`，**不读 `elevation_km`**；
  与一次散射的域一致性是已知边界（高度响应审计时要看）。
- `rt/twilight.py` + `rt/mc.py`：深暮光球面 Monte Carlo，split estimator（一次散射
  确定性、二次以上 MC）。引擎耗时按光子数线性。
- `aerosol.py` / `mie.py`：Mie 光学是 cold-start 的主要开销。`max_cached_wavelengths`
  默认 256，覆盖 81 波长工作网格；预设气溶胶通过 `shared()` 跨 `Atmosphere` 复用，
  因此 warm / 重复调用的 Mie 预处理接近零。冷启动仍需向量化/LUT（后续任务）。
  共享实例是**只读契约**，不要 mutate 气溶胶属性。
- `validation/`：对照公开文献的独立检查。`results.json` 是回归护栏，物理数字变化要
  有独立证据并重跑对应 check。

## 重要约束

### DO NOT

- 不要把 `pip install` 进 `.venv`；依赖变更走 `pyproject.toml` + `uv lock`。
- 不要重做 M10 太阳弦长修复（`geometry.path_lengths_sun` / `refraction.sun_path_lengths`
  的 `up + 2*down`）。
- 不要用关闭折射来“修”暮光；`refraction=False` 只用于复现旧报告数字。
- 不要为了跑得快把 `n_moments` 默认降到 16 而不检查 aureole / dust。
- 不要把“高度响应反号”当作待修的构造问题：`at_altitude` 已改一致截断
  （确定性 1.21×/引擎 1.01×，数据 0.80×，仍反号），残差是低层多次散射缺口。
- 不要在没有独立证据时改动 `validation/results.json` 或弦长修复的物理数字。

### DO

- 使用 `scratchpad/` 目录管理开发任务（详见 `scratchpad/common.md`）。
- 改动物理/数值后跑全量 `pytest`，并重跑受影响的 `validation/check*.py`。
- 文档里的数字与代码行为保持一致；已知偏差要写清楚来源与量级。

## 关键文件

- `docs/STATUS.md` — 一页现状（先读这个）
- `docs/ROADMAP.md` — M1–M10 与 open questions
- `docs/REPORT.md` — 完整报告；`docs/结论.zh.md` — 中文摘要
- `docs/TWILIGHT.md` — 深暮光引擎与引擎基准
- `validation/SECOND_DATASET.md` — Koomen 第二数据集与高度响应
- `skycolor/atmosphere.py`、`skycolor/geometry.py`、`skycolor/refraction.py`
- `skycolor/rt/single.py`、`skycolor/rt/multiple.py`、`skycolor/rt/twilight.py`
- `validation/check5_twilight.py`、`validation/check6_second_dataset.py`
- `scratchpad/HANDOFF_2026-10-30_twilight-perf.md` — 当前交接文档（本地，不进 git）

## 代码风格

- Python 3.12，numpy 向量化优先；避免 Python 循环处理波长/方向块。
- 行宽约 100 列；docstring 与注释用英文，解释“为什么”和物理含义，不只描述代码。
- 公开 API 用类型标注；内部数组形状在 docstring 里写清楚（`(n_view, n_wl)` 等）。
- 新增物理量时，优先在 `validation/` 里留一个可复现的独立检查。
