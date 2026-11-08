# 天空颜色：当前模型与可复算路径

当前生产树（2026-11-08）以一个请求入口保留有限的近似支路，而不是只剩一种模型，
也不是任意物理开关组合。质量与六项物理检查的当前结论统一维护在
[STATUS](STATUS.md)，历史条件和数字留在各专题报告。

## 一条生产流程

```text
Medium + Scene + 太阳折射前坐标
  → render.solve_dome（内部一次rt.solver.solve，四种有限近似）
  → SkyResult（谱、分量、有效设置、诊断）+ SkyPresentation（显示绑定）
  → render_result / chart_table（零求解展示）
```

`solve_dome`绑定网格/地面照明/表观太阳信息；CLI与库共享太阳解析，时间输入不再双重折射。
光谱单位W m⁻² sr⁻¹ nm⁻¹；经CIE XYZ、可选色适应与tone mapping生成图像。
曝光、白平衡不能改变物理谱，也不能替代准确度证据。

大气是US1976空气、O₃/NO₂及Mie气溶胶廓线；太阳侧默认折射，所有支路共用
[生产太阳源](physics/SOLAR_SOURCE.md)。球面一次项仍沿几何直视线积分。
高阶可以是平面SOS或球面直线MC；平面一次项来自同次SOS。
完整的共同目标、支路图、实际数值和归因限制见
[近似说明](physics/APPROXIMATIONS.md)，API/拒绝契约见[架构](ARCHITECTURE.md)。

均匀云deck仅是既有hybrid上的delta-Eddington扩展；水平气溶胶变化只进入声明的
一次项。全阶曲线传播、偏振、结构化三维云、冰晶晕尚非生产能力。
装饰太阳盘不表示光度准确的太阳成像，本轮代表图不画它。

## 三条能照做的读者路径

1. **使用**：[根README](../README.md)给出当前3°场景完整CLI命令；
   [图录](../figures/README.md)列采用版本、场景、离散和共同曝光。
2. **解释**：运行`uv run python -m experiments.approximations.compare`，
   用[薄示例](../experiments/approximations/README.md)的NPZ/CSV复算光谱、Lv与xy；
   从同一结果区分“加高阶”和“换传播近似”。既有数组可以不再求解直接重画。
3. **研究**：[主题入口](../experiments/README.md)保存现象脚本和局部测试；
   [退休索引](../experiments/retired/README.md)保存固定历史库/命令和旧图，
   不再让旧注入工具追随当前内核。生产及最小验证不依赖研究目录存在。

## 已经改善的边界，仍未解决的问题

- 海拔用`at_altitude`一致截断，不重标定柱量；不透明底部与太阳遮挡同源。
  AOD550固定真实550nm光学，不因查询网格改变介质。两项修复的独立证据见
  [海拔审计](twilight/TWILIGHT_ALTITUDE_AUDIT.md)、[网格审计](twilight/AEROSOL_GRID_AUDIT.md)。
- 统一有限日面源已经进入生产；连续漫射折射仍是研究。旧“统一源未进生产”叙述不再适用。
  历史[连续传播报告](twilight/CURVED_TRANSPORT_AUDIT.md)的不同预算对子不能认证纯曲率效应。
- 本轮同入口两场景与三个固定MC种子说明了调用链和分量差别，不认证观测准确性。
  [深暮光参考](twilight/DEEP_TWILIGHT_REFERENCE.md)在正式预算下仍为
  budget-limited-unresolved；很小stderr、拟合改善或零检测不能证明尾部覆盖。
- 回归测试通过不等于物理检查通过。地平线亮度、太阳aureole、绝对暮光验证仍有缺口；
  跨站比值混合介质、季节、频带与太阳轴定义，不能用作纯海拔目标。
  观测用途审计见[参考量](twilight/TWILIGHT_REFERENCE_AUDIT.md)与
  [匹配观测](twilight/MATCHED_OBSERVATIONS.md)。

更容易解释/使用的依据是少了太阳坐标、默认后端、显示链和历史版本的语义猜测，
而不是声称图更像真天空。最终审美仍由人判断；后续物理/性能工作见
[ROADMAP](ROADMAP.md)。环境与命令见[RUNNING](../RUNNING.md)。
