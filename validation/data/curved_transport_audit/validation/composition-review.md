# Continuous transport × unified solar source：组合审读

2026-11-01；审读 `tools/continuous_curved_mc.py` 与 `tools/unified_solar_transport.py`。
本轮只读这两个实现，不改代码；结论是机制审读，不代表阶段四 MC 已统计收敛。

## 组合契约

`mc_solar_source` 从原始 `mc._trace_wavelength` 取源码，先替换 solar NEE，再调用
`transport(code, source.atmo)`；只编译一次。连续 transport 的 replacement 将
`source.scatter(... da[ci] ...)` 和 `_rotate(da[si], ...)` 分别改为 `arrival[ci]`
和 `arrival[si]`。因此相函数与散射抽样都围绕碰撞点的入射方向，不沿用自由程出发方向。
自由程与散射随机数的数量/调用位置不增加；曲线几何改变碰撞后的接受/存活序列属于物理变化。

`FiniteSolarSource.validate_transport` 要求同一个 Atmosphere、相同 medium/scene 对象及完整
波长网格，防止 MC 内部截取波长后把局部索引错误用于全谱太阳表。

## 全局 majorant 为什么合法

连续 flight 仅暴露 ground/top 一层，原 tracer 的 `major[0]` 因而成为所有 fine-grid
`beta_ext` 节点最大值乘 1.005。tracer 接受概率读的是这些节点定义的 log-linear 场；
log-linear 正函数在每段的最大值位于端点，因此节点最大值严格控制整个实现的消光场。
沿真实弧长抽样 `s=-log(U)/major[0]`，再沿连续折射轨迹推进，构成合法的 delta tracking。
不需处理曲线跨越旧 coarse 层之后 majorant 变化的问题。

这里保证的是 **MC 所用 fine-grid 插值场**，不是未经插值的任意连续密度峰值。
任意亚网格窄 aerosol layer 仍需细化光学表；这是原 MC 的离散化边界，不是 RK majorant 漏项。

## n²、太阳源与地面

传播的变量是 reduced radiance `L/n²`，因此无吸收自由程不加权。太阳表的局部增益为
`G_local=(n_vertex/n_inf)² dOmega_app/dOmega_true`。源函数额外乘
`(n_observer/n_vertex)²` 后，参考到检测器的增益恰为
`(n_observer/n_inf)² dOmega_app/dOmega_true`，没有重复或丢失顶点的 n²。

同一个顶点的散射算子两侧具有同一 n，因此 phase 与 albedo 不再加 n²。
Lambertian 地面也在同一局部介质内，反射 continuation 的 throughput 仍乘 rho；
`source.ground` 中的太阳照明已由上述源转换参考到 observer，所以地面分支无需另乘 n²。
忽略真实折射率界面的 Fresnel 损失与当前连续大气/漫反射模型一致。

## 发现的继承数值边界：散射混合物分别 log 插值

原 MC 同时独立 log 插值 `beta_sca=sum(beta_species)` 和各个 species。
通常 `loginterp(sum(beta_species)) >= sum(loginterp(beta_species))`。
NEE 中 albedo 与 mixture 分母消去，因此 NEE 使用 component sum；continuation 却按总量
albedo 存活，然后归一化 component fractions，所以其高阶散射率会略高于 component sum。
这是原 MC 已有的微小离散化不一致，不影响上面 extinction majorant 的合法性。

独立探针直接调用 `mc._fine_grid`、`mc._aerosol_spectral`，对每段 midpoint 比较
`sqrt(S_i*S_j)/sum(sqrt(component_i*component_j))-1`；clear 81 波长的三个代表波长：

| ground km | wavelength nm | 最大相对 excess | 所在高度 km |
|---|---:|---:|---:|
| 0 | 450 | 3.3803e-5 | 3.0188–3.0894 |
| 0 | 550 | 4.5848e-5 | 3.6065–3.6837 |
| 0 | 650 | 5.5418e-5 | 4.0816–4.1637 |
| 2.4 | 450 | 1.3054e-5 | 4.6083–4.6684 |
| 2.4 | 550 | 2.0270e-5 | 4.8535–4.9168 |
| 2.4 | 650 | 2.6930e-5 | 5.1781–5.2454 |

这些是局部插值差，不是最终 radiance 偏差上界。当前百分比级统计噪声下不值得为它扩大投入；
若目标转为 1e-4 精度，应把 fine-grid refinement 与 RK refinement 分开，并统一 component
sum 的权威实现。确定性 SS 使用连续密度而 MC 使用 log-grid，也因此没有逐位闭合的承诺。

## 曲线边界的适用边界

RadialSolarMapping 验证 `q(r)=n(r)r` 单调，因此真实轨迹仅能有一个径向 minimum，不能有
径向 maximum。Hermite 子步显式检查 down→up 的 minimum，防止一次子步穿地又出地。
这建立了当前标准 profile 的边界机制；不构成任意粗步长、任意剧烈自定义 n 的误差保证。
独立 4/2/1 km 轨迹与掠地探针的证据见 `continuous-probe.json`，最终 MC 仍需分步长检验。

## 最敏感的统计/收敛场景

1. ground 0 与 2.4 km 均保留：低层水平漫射、ground hit 与近切线自由程对 curvature 最敏感。
2. sun=-6° 的 zenith 是阶段三对照；接近地平线的 view 更放大累计弯曲与掠地边界误差。
3. 有限太阳盘跨 shadow edge 的 source vertices，应看 4/2/1 km 的统计变化，不能仅看
   终点方向误差小；小几何偏差可能改变是否能见太阳盘。
4. 各独立种子应先做同一 transport 的步长收敛，再比较 straight/curved；不要把更换
   global majorant 所改变的随机路径误作曲率贡献。2 km freeflight 性能不能代替整条 MC 性能。

当前未发现需要阻止组合实验的新增正确性缺口；以上边界限定结论范围，阶段四数值结论留给主线。
