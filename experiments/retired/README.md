# experiments/retired — 退休研究主题索引

本目录只承载**冻结主题的说明与历史图**；代码不在 main 活跃树中，不承诺适配当前
生产内核。复跑一律在固定历史树进行：`git worktree add <dir> <SHA>` 后按该树的
`uv.lock` 建环境，用下表逐字命令。历史代码与结论的长期保全（tag/分支）由 owner 负责。

## 冻结引用（两级版本语义）

**代码冻结 SHA** = 退休代码最后的主树副本（读代码用）。**复跑库 SHA** = 该工具链
实际可运行的匹配库版本——两者不同：`afaae17` 的 `skycolor/rt/mc.py` 已被 task-6
整合提交 `a15d87c`（merge `6bdb1f8`）重写，源码注入替换在其文本上
`compile` 即 `SyntaxError`（line 193，2026-11-08 实测复现）；而审计基线 `5ac0c74`
的 mc.py 文本上同样替换**编译通过**（同日实测）。不要用代码冻结 SHA 复跑注入链。

| 引用 | SHA | 角色 |
|---|---|---|
| 代码冻结基线 | `afaae17` | 退休 tools/experiments 的最后主树副本（仅读代码） |
| 太阳源/注入链复跑树 | `5ac0c74` | 审计基线 main：含全套审计期工具与数据包；`unified_solar_transport` 的源码替换在该库文本编译通过（实测） |
| 曲率复跑 checkpoint | `2756b2f` | 包 README 记录的 checkpoint；运行时源 = 该源 + `source-binding-guard.patch` 反向应用（包内 hash 验证） |
| task-7 深暮光 | `dda8673` | 执行器/汇总器/二次候选与原内核（本任务在自身固定树 `--verify-only` 全绿实测） |

## 曲率传播（curved transport audit）

- 代码（@ `afaae17`）：`tools/check_curved_hamiltonian.py`、`check_curved_single.py`、
  `check_majorant_equivalence.py`、`continuous_curved_mc.py`、`curved_mc_transport.py`、
  `summarize_curved_transport.py`
- 数据：`validation/data/curved_transport_audit/`（在位冻结，含源码快照
  `snapshots/*.py`、`source-binding-guard.patch`、hash manifest）
- 复跑：**checkpoint `2756b2f`** 的固定树 + 将 `source-binding-guard.patch` 反向应用
  （还原运行时源，包内 `runtime-source-provenance.json` hash 验证），再按该树
  `validation/data/curved_transport_audit/README.md` 的命令执行；已有可信长预算记录
  （`logs/`、`raw/*.jsonl`），不重启 ensemble。
- 已知限制：只在其记录的 checkpoint+patch 组合成立；不随当前 `rt/mc.py` 运行。

## 太阳源审计链（solar source audits）

- 代码（@ `afaae17`）：`tools/audit_solar_source.py`、`audit_single_solar_source.py`、
  `point_solar_table.py`、`radial_solar_mapping.py`、`continuous_solar_columns.py`、
  `unified_solar_transport.py`、`check_solar_disk_mc.py`、`check_solar_etendue.py`、
  `probe_solar_refraction.py`、`run_unified_solar_audit.py`、`summarize_solar_source.py`、
  `summarize_unified_source.py`、`summarize_bandpass_audit.py`
- 复跑树：**`5ac0c74`**（审计基线 main）。`afaae17` 的 mc.py 已含 task-6 重写，
  注入替换在该文本编译失败（`SyntaxError`，实测）；`5ac0c74` 编译通过（实测）。
- 数据：`validation/data/solar_source_audit/`、`unified_solar_source_audit/`、
  `twilight_reference_audit/`（在位冻结；源表与源码 hash 可回查）
- 生产采用证据（不受退休影响）：`validation/production_solar_source.py --suite source`
  真实读取 `solar_source_audit/geometry-probe.json` RK 锚对生产 `SolarRayField`；
  生产实现文档 `docs/physics/SOLAR_SOURCE.md`。
- 已知限制：注入机制（`inspect`→替换→`exec`）只在对应历史内核成立。

## 暮光海拔审计（twilight altitude audit）

- 代码（@ `afaae17`）：`tools/twilight_altitude_audit.py`、`summarize_twilight_altitude.py`
- 数据：`validation/data/twilight_altitude_audit/`（在位冻结，含 Koomen1952 对）
- 结论入口：`docs/twilight/TWILIGHT_ALTITUDE_AUDIT.md`；残差物理根因未定（AGENTS.md）。

## 重复展示编排（gallery / timelapse）

- 代码（@ `afaae17`）：`experiments/06_gallery.py`、`experiments/07_timelapse.py`
- 退休理由：重复展示编排；不为一张候选图维护完整画廊/动画工作流。
- 旧图去向：`06_gallery_*.png`、`06_hero_*.png`、`07_sunset.gif` 留在根 `figures/`
  （生产候选池，任务4甄别）；`07_sunset_ends.png`（GIF 首尾帧中间图）在本目录 `figures/`。

## 历史廓线与深暮光图（profiles / twilight figures）

- 代码（@ `afaae17`）：`experiments/12_profile_sensitivity.py`、`profile_variants.py`、
  `twilight_figure.py`、`twilight_nocturnal.py`
- 已知限制：`twilight_figure.py` 硬编码旧 M10 数值，其输出不能当作当前模型精度图。
- 生产侧真实保障另存：默认套件 `tests/test_profiles.py`。
- 旧图去向：`12_profile_sensitivity.png`、`11_twilight_deep.png` 在本目录 `figures/`；
  `14_twilight_altitude_audit.png`、`15_solar_source_audit.png`（审计专题）同。

## 本目录 figures/

| 图 | 内容与限制 |
|---|---|
| `07_sunset_ends.png` | timelapse GIF 首尾帧中间图 |
| `10_cloud_albedo_and_sunset.png` | 验证与生产混合图；整体按历史验证/研究用途保存，任务4需要时拆选重画 |
| `11_twilight_deep.png` | 深暮光引擎比较（旧 M10 硬编码数值 + 统计限制），ROADMAP M19 |
| `12_profile_sensitivity.png` | 历史廓线敏感性（`docs/twilight/PROFILES.md` 冻结期有效） |
| `14_twilight_altitude_audit.png` | 海拔审计图（域截断语义曾变更，不作当前海拔结论） |
| `15_solar_source_audit.png` | 太阳源审计研究图（生产采用证据在 `validation/production_solar_source`） |

## task-7 深暮光参考（deep twilight reference）

- 全部代码（@ `dda8673`）：`tools/deep_twilight_reference.py`（执行器，绑旧研究源栈
  `point_solar_table`/`unified_solar_transport`）、`summarize_deep_twilight_reference.py`
  （汇总器，`--verify-only` 只读归约）、`twilight_second_order.py`（二次候选及其测试）、
  `skycolor/rt/mc.py` 诊断 sink（opt-in，默认关）、`tests/test_deep_twilight_reference.py`、
  `tests/test_twilight_second_order.py`
- 数据：`validation/data/deep_twilight_reference/`（在位冻结；manifest 锚定。
  已知偏差：main 合入版 `README.md` 为 salvage 合并改写版，与 manifest 固定 hash 不同；
  数据文件 81/82 逐位一致，见任务 `verification/baseline.md`）
- 复跑（固定树 `dda8673` 内，本任务已实测全绿）：

  ```bash
  uv run python tools/summarize_deep_twilight_reference.py \
      --directory validation/data/deep_twilight_reference --verify-only
  ```

- 结论：两深度 budget-limited-unresolved；−9° 三锚点源修正识别；二次候选混合边际
  （`docs/twilight/DEEP_TWILIGHT_REFERENCE.md`、ROADMAP M19）。
- 已知限制：汇总器/执行器与 `dda8673` 内核绑定（诊断 sink 未入生产 mc.py）；
  二次候选未升为生产近似，不以合成自闭合声称正确。
