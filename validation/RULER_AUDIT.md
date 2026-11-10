# RULER AUDIT — check2 / check3 / check6 对照合同（尺子）审计

2026-11-10。task-ruler-audit 交付物。只读审计：本文所有裁定不改动
`skycolor/`、`validation/results.json` 与既有 check 判据——修尺子是审计结论
产出后的独立任务（owner 裁定立项）。

审计对象：三个 check 的**对照合同**——参考标准的大气/浊度前提、物理量定义、
太阳轴约定、数值实现/转录，与被对照的模型场景是否一致。被审计的是尺子，
不是被测的模型。

证据等级约定：

- **原文前提核对**：前提读自原始出版物（或其归档全文审计
  `docs/twilight/TWILIGHT_REFERENCE_AUDIT.md`，含来源 hash）；
- **复算**：本文档数字可由 `uv run python -m validation.ruler_audit_recheck`
  从提交在库的数据/模型独立再生；
- **复现（二级）**：前提经公开的第二手复现核对（原文付费墙时使用，逐处标注）；
- **推断**：推断，明确标注。

## 裁定矩阵总表

4 前提维度 × 3 check。每格：裁定（成立 / 不成立 / 不可裁决）+ 证据等级 +
一句话依据。矩阵格只声明实际核查范围；未覆盖子维度写「未核 + 理由」，不默示成立。

| 前提维度 | check2（CIE 110 DHI / G173 / Bird） | check3（CIE 110 相对分布） | check6（Koomen 1952） |
|---|---|---|---|
| 1. 大气/浊度前提 | 待填（Step 2） | 待填（Step 3） | 待填（Step 4） |
| 2. 物理量定义 | 待填（Step 2） | 待填（Step 3） | 待填（Step 4） |
| 3. 太阳轴约定 | 待填（Step 2） | 待填（Step 3） | 待填（Step 4） |
| 4. 数值实现/转录 | 待填（Step 2） | 待填（Step 3） | 待填（Step 4） |

## 模型侧场景统一陈述

（三个 check 被对照的模型侧场景，复算与报告统一引用；数字可复算。）

- 太阳源：统一有限盘太阳源为生产默认——**真高度（true elevation）入参**，
  内部折射表映射视方向，盘面为均匀 cap、半径 16′（`skycolor/solar_source.py`，
  `docs/physics/SOLAR_SOURCE.md`）。
- 求解器：check2/check3/−6° 走 `sky_radiance_total`（单次 + SOS 多次散射，
  hybrid-sos；`validation/common.py` 默认 `SKYCOLOR_MULTIPLE=1`）；−9°/−12° 走
  球面引擎 `twilight_radiance(quality="low", seed=0)`（split estimator，单 seed）。
- 地面：Lambertian，`ground_albedo` 默认 0.1，进 SOS 漫射场
  （`skycolor/atmosphere.py` 构造参数、`skycolor/rt/multiple.py`）。
- 场景参数：check2/check3 默认 `clear` preset（Hinterland 气溶胶、AOD₅₅₀=0.05、
  O₃=300 DU、海平面）；check2 的 G173 段用 `aod550=0.0734, ozone_du=343.8`。
- 波长与积分：光度积分 K_m=683，CIE 1931 2° ȳ，网格 `WL_COARSE`（380–780 nm,
  5 nm）；DHI = 光亮度 × cosθ 半球积分（`validation.common.horizontal_illuminance`）。

## check2 — CIE 110 DHI / G173 / Bird 合同审计

（Step 2 填充：2a G173 前提、2b Bird 转录、2c CIE 110 DHI 合同、2d 地面反照率。）

## check3 — CIE 110 相对分布合同审计

（Step 3 填充：3a 公式保真、3b 太阳轴、3c 归因证据、3d 适用前提、3e 反照率覆盖。）

## check6 — Koomen 1952 观测合同审计

（Step 4 填充：4a 转录比对、4b 物理量定义调和、4c 站点透过率、4d 太阳轴、
4e 分支区分、4f 反照率覆盖。）

## FAIL 分流表

（Step 5 填充：「修正对照条件后可能改判」vs「指向真实物理缺口」，每行可追溯
矩阵格与证据。）
