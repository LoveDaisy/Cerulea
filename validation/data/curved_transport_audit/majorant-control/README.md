# Coarse/global majorant：机制、历史差额与新增控制

完整审计见 [majorant-equivalence-audit.md](majorant-equivalence-audit.md)。
本目录不新增MC结果；原始记录仍位于上一级`raw/`以及阶段三的冻结资料。

- `historical-majorant-equivalence-audit.md`、`historical-majorant-equivalence-probe.py/.json`：
  追加新控制以前的报告、执行脚本和输出，保留高站跨批−2.244SE的历史证据。
- `majorant-equivalence-audit.md`：完整报告，追加新native seed310–317控制。
- `majorant-equivalence-probe.json`：永久工具重算的mechanism/dense bounds、原对比与新控制。
- `snapshots/aerosol_596aaba.py`、`mc_596aaba.py`：阶段三生产基线；独立重执行旧Mie归一
  方法，确认full81光学量不受随后550nm归一修复影响；MC源码则逐字未变。
- `snapshots/unified_solar_transport_7ed71a62.py`：SHA与phase03 raw精确相符的源实现。
- `snapshots/unified_solar_transport_before_guard.py` / `after_guard.py`：phase04 global
  与新native运行时版本；SHA分别5d25bf55…/6ed1b03c…。精确差异仅为六行构造前置检查。
- `snapshots/continuous_curved_mc_runtime.py`：冻结的global/continuous flight工具版本。
- `source-binding-guard.patch`、`runtime-source-provenance.json`：guard-only差异及来源。
- `inputs.sha256.json`：17个冻结输入，含相对引用的原始光谱和control摘要；派生的新JSON
  与更新后的报告不参与循环校验。工具另核phase03既有22份冻结输入。

复跑只读机制与统计审计，约1s，不运行新光子：

```bash
uv run python -m tools.check_majorant_equivalence
```

默认根目录取工具`__file__`，默认全部使用tracked资料，无scratchpad或git查询。
`--directory`可指向另一份curved_transport_audit，`--source-data`可指向另一份phase03档案，
`--out`可将派生JSON写到指定路径。
`portability-check.json`记录两项实际复验：切换cwd后默认资料仍复现，整套curved/phase03
档案复制到随机临时目录后也复现相同JSON；均不读取git历史或scratchpad，不产生新光子。

结论边界：两高度/全81的majorant无漏界，光学、source、50阶cap和散射/RR相同；native
0.1mm内部穿层位移仍是明确数值差异。历史−2.244SE保留，新native−global为
.00318506±.00365615cd/m²（+0.871SE），后者是seed批次配对而非相同光子路径。
这些证据没有确证majorant系统偏差，也不证明稀有尾部及所有系统误差已收敛。
