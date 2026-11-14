# AGENTS.md

## 项目概述

物理基础的光谱天空颜色模拟器。从太阳光谱、Rayleigh/Mie 散射、臭氧吸收和球面辐射传输
第一性原理计算天空光谱辐亮度（W m⁻² sr⁻¹ nm⁻¹），再经 CIE 色度学 → 色适应 →
tone mapping 得到颜色与图像。Python 3.11+（开发基线 3.12），numpy / scipy，环境由 uv + `uv.lock` 固定。

## 常用命令

### 环境

```bash
uv sync        # 从 uv.lock 重建 .venv；不要 pip install 进环境
```

### 测试

```bash
uv run pytest -q                      # 默认 = 快层 dev loop（389 项，~38-63s 窗载依赖，
                                      # 12-worker xdist / LPT（longest-processing-time
                                      # 优先调度）；slow 标记项被
                                      # 过滤，显式 -m/-k/::nodeid 退让）
SKYCOLOR_FULL_SUITE=1 uv run pytest -q -n 12 --dist=loadgroup
                                      # 全层验收入口（582 项全人口，零 skip）
SKYCOLOR_FULL_SUITE=1 uv run pytest -n 0 -q tests/test_validation_harness.py
                                      # 结构测试串行诊断（env=1 全量含 slow 成员）
uv run python -m validation.run_all   # 六项物理校验，单池 8 worker，实测约 4 min；
                                      # 既有物理 FAIL 预期 exit=1（不经 pytest，
                                      # 与分层无关）
```

双层契约与定层规则见 `docs/DEVELOPMENT.md`「Test tiers」；计数链、形态实测、
启动暂态与 formal pair 协议（显式全层）见 `validation/FAST_SUITE.md`
suite-tier 节。新增测试由作者写时定层（种类判据 + 成本估算，`kind:`/`cost:`
reason）；marker 为人工声明（升线自动闸已按 2026-11-11 用户裁定删除），
成本经权重再生成流程落表（`fast_predicted_wall_s` 记录性指标，无自动闸）。

### 运行

```bash
# 渲染一张图
uv run python -m skycolor.cli render --sun-elevation 3 --camera perspective \
    --size 1200x520 --pitch 14 --hfov 100 --preset clear --aod 0.05 \
    --yaw 180 --out sunset.png

# 生成实验图（写各主题目录 figures/，仓库根目录运行）
uv run python -m experiments.approximations.compare
# 近似/实测解释：docs/physics/APPROXIMATIONS.md；正式图/完整参数：figures/README.md
```

### 代码格式化

项目没有 linter / formatter 配置；沿用现有风格（见文末“代码风格”）。

## 开发与合并惯例

- 代码、文档、配置改动均先基于当前 `main` 建独立主题分支，在分支内修改、验证并提交。
- 合并前用 `git diff main...HEAD` 核对范围，再在主题分支执行 `git rebase main`；
  回主仓以 `git merge --no-ff <branch>` 合入，保留明确的任务边界。
- 合并后核对树内容与工作区状态；独立任务并行时用隔离 worktree。
- 具体操作、按改动类型选择测试及物理校验判读见 `docs/DEVELOPMENT.md`。

## 架构与设计

完整模块树与模块契约的唯一权威在 `docs/ARCHITECTURE.md`（本文件不再维护副本树，
避免入口文档与契约漂移）。速览：

- **生产入口**：`skycolor/rt/solver.py::solve`（唯一权威组装入口，四近似组合契约）；
  展示层 `render.solve_dome`；CLI `skycolor/cli.py`（render / chart）。
- **物理核**：frozen `Medium`（`medium.py`）+ frozen `Scene`（`scene.py`，含唯一太阳
  输入解析器 `resolve_sun`），`Atmosphere` 是 facade；后端共享核 `solar_source.py`
  （统一有限太阳盘源）；`rt/` 下 single / multiple / mc / twilight / cloud / result；
  `spectra/`、`mie.py`、`aerosol.py`、`cloud.py`、`geometry.py`、`refraction.py`、
  `optics.py`、`constants.py`（物理常数与工作波长网格的单一权威）、`color.py`、
  `render.py`、`presets.py`、`data/`。
- **外围**：`validation/`（check1..6 + common.py）、`experiments/`（现象主题 +
  cssp/retired）、`tests/`、`docs/`、`tools/`（build_data / perf_bench /
  make_scheduling_weights / make_mie_anchor_arrays 窄维护）、`figures/`、
  `data_raw/`（数据源文件，`skycolor/data/SOURCES.md` 逐一定向）、
  `scratchpad/`（本地任务管理，gitignore）。

### 核心模块说明

- `atmosphere.py`：`Atmosphere` 是 facade：物理介质在 frozen `Medium`（`skycolor/medium.py`），
  观察场景在 frozen `Scene`（`skycolor/scene.py`）。`at_altitude(z0)` 已改为一致截断
  （`z_bounds` 自 z0 起、地面=z0、不重标定任何柱量）。后续审计发现太阳遮挡
  仍按海平面，已改为 single/SOS/MC/cloud 均用 `r_bounds[0]`；−6° 天顶确定性高低站比
  由历史 1.208× 变为 1.027×。旧引擎 1.01× 也属修复前记录，当前结果及参考条件见
  `docs/twilight/TWILIGHT_ALTITUDE_AUDIT.md`；剩余差异的物理根因尚未确定。
- `geometry.py` / `refraction.py`：太阳弦长是暮光的关键。M10 修复后（M 编号注册表见
  `docs/ROADMAP.md`），太阳在局部水平以下时走“远侧 perigee→space + 近侧 perigee→point”
  （`up + 2*down`）。不要重做。
- `rt/solver.py` / `rt/result.py`：`solve` 是唯一权威组装入口（请求解析、能力闸、核调度、
  溯源记录，四近似组合契约）；`SkyResult` 是溯源契约。
- `solar_source.py`：统一有限太阳盘源（n(r) 场、连续折射柱量、几何束），全部后端共用。
- `rt/single.py`：球面几何精确一次散射，从观察者半径 `r0` 开始积分；照明来自统一太阳源
  `SolarSource`。`RefractionGeometry` 已降级为遗留数据管线/测试锚角色（生产零消费者；
  `tools/build_data.py` 打包 `skycolor/data/refraction.csv` 仍用 refraction 模块）。
- `scene.py` / `constants.py`：frozen 观测场景 + 唯一太阳输入解析器 `resolve_sun`
  （CLI 与库共用，防双折射）；物理常数与工作波长网格（`WL_DEFAULT`/`WL_COARSE`）的
  单一权威，下游 import 而非重推导。
- `rt/multiple.py`：平面平行漫射场，地面固定在 `z_bounds[0]`，**不读 `elevation_km`**；
  与一次散射的域一致性是已知边界（高度响应审计时要看）。
- `rt/twilight.py` + `rt/mc.py`：深暮光球面 Monte Carlo，split estimator（一次散射
  确定性、二次以上 MC）。引擎耗时按光子数线性。
- `optics.py::contiguous_transpose`：macOS arm64 numpy 2.5.4 wheel 的系统 Accelerate
  dgemm 对「第二操作数 F-contiguous 且内积维 k=3」越界读（读不改值但间歇 SIGSEGV，
  libgmalloc 下必现）。skycolor 所有 `X @ Y.T`（k=3）站点必须经它路由：m≥64 两内核
  逐位相同（golden 逐位不变；results.json 模型字段逐位不变，唯一例外是 check6 派生
  tolerance 字段末位 1 ulp——非物理字段），站点清单与契约由
  `tests/test_blas_layout_guard.py` 机械拦截。根因与逐位证据链在本地任务日志
  `task-mc-restrict-segv`（scratchpad，gitignore；结论由守卫测试与 results.json
  基线承载）。
- `aerosol.py` / `mie.py`：冷启动已批式向量化（本地任务日志 `task-cold-optics`，
  untracked）：系数核与标量
  路径逐位等价（CPython 除法站点按 Smith+FMA 常量提升复现），矩/表核为 D2 声明
  容差（ulp 级，锚测试钉死）；冷 dusty `delta_m(32)` 实测 39.4→3.1 s（~13×）。
  `max_cached_wavelengths` 默认 256，覆盖 81 波长工作网格；预设气溶胶通过
  `shared()` 跨 `Atmosphere` 复用，warm / 重复调用的 Mie 预处理接近零。填充按
  请求 λ 集分组批，状态保留逐 pair 精确长度系数（内存轮廓同标量）。
  共享实例是**只读契约**，不要 mutate 气溶胶属性。
  AOD550 的参考截面固定为实际550 nm状态；查询网格和 `Atmosphere._restrict`
  不得重新归一介质。单波长旧归一错误已修复，见 `docs/twilight/AEROSOL_GRID_AUDIT.md`。
- `validation/`：对照公开文献的独立检查。`results.json` 是回归护栏，物理数字变化要
  有独立证据并重跑对应 check。

## 重要约束

### DO NOT

- 不要把 `pip install` 进 `.venv`；依赖变更走 `pyproject.toml` + `uv lock`。
- 不要重做 M10 太阳弦长修复（`geometry.path_lengths_sun` / `refraction.sun_path_lengths`
  的 `up + 2*down`）。
- 不要用关闭折射来“修”暮光；`refraction=False` 只用于复现旧报告数字。
- 不要为了跑得快把 `n_moments` 默认降到 16 而不检查 aureole / dust。
- 不要恢复 `at_altitude` 的双重缩放或用海平面半径判断高站太阳遮挡；域截断与不透明
  地面遮挡必须同源于 `r_bounds[0]`。不要把历史 1.21×/1.01× 当当前结果，或把跨文献
  0.80× 当受控海拔目标；残差不能直接归因为低层多次散射不足。
- 不要在没有独立证据时改动 `validation/results.json` 或弦长修复的物理数字。
- 不要在 skycolor 生产代码新增 `X @ Y.T`（k=3）matmul 而不经
  `optics.contiguous_transpose` 路由（机制、根因与逐位证据见「核心模块说明」的
  optics 条目；`tests/test_blas_layout_guard.py` 会拦截违规站点）。

### DO

- 使用 `scratchpad/` 目录管理开发任务（详见 `scratchpad/common.md`；该目录 gitignore，
  手册只在本地 checkout 存在，fresh clone 上不可达）。
- 改动物理/数值后跑全量 `pytest`，并重跑受影响的 `validation/check*.py`。
- 文档里的数字与代码行为保持一致；已知偏差要写清楚来源与量级。

## 关键文件

- `docs/STATUS.md` — 当前行为与验证边界（先读这个）
- `docs/README.md` — 文档分工、当前入口、暮光审计顺序与历史报告定位
- `docs/ROADMAP.md` — 已完成里程碑、验证依据与 open questions
- `docs/DEVELOPMENT.md` — 分支、提交/合并惯例和开发/测试流程
- `docs/REPORT.md` — 完整报告；`docs/SUMMARY.zh.md` — 中文摘要
- `docs/twilight/TWILIGHT.md` — 深暮光引擎与引擎基准
- `validation/SECOND_DATASET.md` — Koomen 第二数据集与高度响应
- `skycolor/atmosphere.py`、`skycolor/geometry.py`、`skycolor/refraction.py`
- `skycolor/rt/single.py`、`skycolor/rt/multiple.py`、`skycolor/rt/twilight.py`
- `validation/check5_twilight.py`、`validation/check6_second_dataset.py`
- 暮光审计 M13–M18：专题与证据入口统一见 `docs/twilight/README.md`，实验后端尚未替换生产默认。
- `scratchpad/`（untracked）— 本地实验/任务记录与临时 session 交接；旧交接已退休，长期成果维护在 tracked 文档

## 代码风格

- Python 3.12，numpy 向量化优先；避免 Python 循环处理波长/方向块。
- 行宽约 100 列；docstring 与注释用英文，解释“为什么”和物理含义，不只描述代码。
- 公开 API 用类型标注；内部数组形状在 docstring 里写清楚（`(n_view, n_wl)` 等）。
- 新增物理量时，优先在 `validation/` 里留一个可复现的独立检查。
