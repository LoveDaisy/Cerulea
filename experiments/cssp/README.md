# experiments/cssp — CSSP 观测输入主题

CSSP（2014 年多伦多黄昏光谱观测）观测输入工具与冻结数据包的拥有目录。
工具 **hermetic**：仅依赖 stdlib + numpy，不 import `skycolor`，不随生产内核演化。

## 入口

```bash
# 离线结构校验（只读，无网络）
uv run python -I experiments/cssp/cssp_observation_inputs.py verify \
    --packet validation/data/cssp_observation_inputs

# 重放链文档化（raw → published，只读）
uv run python -I experiments/cssp/cssp_observation_inputs.py replay \
    --packet validation/data/cssp_observation_inputs
```

## 数据包（冻结，hash 锚定，不移动）

| 包 | 内容 | 锚 |
|---|---|---|
| `validation/data/cssp_observation_inputs/` | 42 条记录：raw MAT/JPL/AERONET/IEM/NOAA、发布谱、星历、标定 | `inputs.sha256.json`（82 项） |
| `validation/data/cssp_observation_feasibility/` | 旧可行性包（atmosphere/instrument/timing） | `inputs.sha256.json`（23 项） |
| `validation/data/matched_twilight_observations/` | 匹配观测包（工具读其星历 knots） | `inputs.sha256.json`（38 项） |

## 测试

```bash
uv run pytest experiments/cssp/tests -q   # 58 项，不进默认生产收集
```

覆盖：观测时间/会话切分、曝光语义、逐位重放（rtol=1e-6 独立容差）、
结构拒绝（非有限/负通道）、force 原子写、半球能流算子解析判据
（πL / cos²h / 闭合方位——CSSP 算子带地平线遮挡语义）。

## 边界与证据

- 冻结已完成范围：不扩展抓取、标定或观测研究；`replay` 的重放一致性**不是**
  独立绝对定标（工具输出原话注明）。
- 任务证据长报告：`docs/twilight/CSSP_OBSERVATION_INPUTS.md`（历史文档，
  其中的 `tools/` 路径为当年有效路径，现以本 README 的命令为准）。
- 生产地面积分的独立保证在默认套件：`tests/test_render.py` 半球解析锚 +
  `tests/test_presentation.py` 消费链锚，不依赖本主题。
