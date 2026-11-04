# 阶段三：统一太阳源的冻结证据

结论及数值边界见 [UNIFIED_SOLAR_SOURCE_AUDIT.md](../../../docs/twilight/UNIFIED_SOLAR_SOURCE_AUDIT.md)。
这是实验资料，不是生产 LUT 或 `validation/results.json` 的替代品。

- `raw/`：48 条未经重写的 JSONL 光谱记录；3 source ×2 ground ×8 seeds。
  原记录中的 scratchpad/绝对路径属于历史 provenance，复跑入口使用下述相对 manifest。
- `tables/`：4 份 full81 NPZ（几何/连续折射 ×海平面/2.4 km）；原始 bytes、label 和
  medium 契约全部保留。Legacy 的 manifest entry 指向同高度折射表，但 legacy evaluator
  使用原库源；这个 entry 不代表它的 SS/MC 已共用该表。
- `tables-manifest.json`：相对于本文件所在目录的路径，可重定位，显式保留原 label。
- `validation/`：独立 ray-tube 能流、连续柱量/常 n、spline knot 反例及分段修正、
  高度/角度 refinement、四表 heldout、SS/disk 收敛、一次闭合及阶段三 pytest 日志。
- `inputs.sha256.json`：22 份冻结输入的 SHA256；不含派生的 summary 与图，避免循环校验。
- `summary.json`、`summary.png`：`uv run python -m tools.summarize_unified_source` 生成。
  所有光谱重新积分 photopic，±仅为8个独立 seed 的标准误；高度比用配对 delta method。

`high-row-derivative-probe.json` 是**历史失败实现**的反例；
`high-row-knot-split-probe.json` 是修正后的 Gauss 收敛，
`oracle-knot-independent.json` 是另一路 DOP853/面积束证据。不要混合两版数值。

复跑不需要 scratchpad。源表 loader 会拒绝不同 label、波长/medium sampled signature；
这个 sampled signature 不证明任意用户 callable 的全域等价，所以这些表只供本冻结 clear 实验。
原始记录基线为 `596aaba`，工具另由各记录 `source_hashes` 定版；它们生成于 aerosol grid
修复之前，但 full81 含550 nm，本批光学系数不受该修复影响。

仅本阶段：solar ray 已连续折射，MC diffuse ray 仍直线。曲线漫射实验不包含在此目录。
