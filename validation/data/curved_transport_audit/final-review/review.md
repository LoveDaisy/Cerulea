# 阶段03/04独立最终机制审读

2026-11-01。只读审查，未修改已有实现或报告，未重跑完整MC。
审读 `unified_solar_transport.py`、`point_solar_table.py`、`continuous_curved_mc.py`、
原MC的传播/NEE/散射组合，以及三份tracked报告：
`UNIFIED_SOLAR_SOURCE_AUDIT.md`、`AEROSOL_GRID_AUDIT.md`、`TWILIGHT_REFERENCE_AUDIT.md`。
核对了阶段03归档summary及原始记录；**不提前核准正在分析的阶段04完整ensemble数值**。

## 结论

在当前明确的标准单像大气、标量非偏振RT、均匀16′太阳盘和局地Lambert地面的范围内，
未发现仍未处理的 source/传播组合阻塞项。发现一个真实的source-table绑定反例，
root已立即修复，我已独立复验拒绝路径；现有实验的加载链没有使用错误绑定。

阶段03的source-only结论可以按报告所列范围签发：统一源定义、太阳能流/柱量证据、
photopic一次项闭合、受控−6°天顶实验的有限预算SS+MS结果。它们不等于生产完整折射
天空已经正确，也不证明真实站点高度效应或剩余绝对误差已被因果解释。

阶段04目前可以签发连续flight的机制、独立轨迹和一次项检查；完整高阶曲率贡献、
majorant control统计相容性及其不确定度，须由主线完成现存ensemble的分析后另签。

## 1. 实际反例：同波长网格不证明同一太阳表介质（已修复）

旧构造只检查`atmo.wl == table.wl`。将ground0源表传给ground2.4的Atmosphere，
`FiniteSolarSource(high_atmo, sea_table)`以及其`validate_transport(high_atmo)`均能通过。
后者保证source使用其原始Atmosphere，并没有证明该source当初绑定了正确的table。

实测反例：相同81λ、查询高度2.4km、真实太阳高度−.45°、550nm：

| 源表 | apparent deg | T550 | G | lit |
|---|---:|---:|---:|---|
| ground0 | −.00766255 | .0170407383 | .85903924 | true |
| 正确ground2.4 | 遮挡占位值 | 0 | 0 | false |

两种表的opaque ground不同，错误绑定可以把应遮挡的方向错误照亮。
**这不是现有归档数据被污染的证据**：`table_for`构建于传入atmo，显式manifest分支
`PointSolarTable.load(... atmo, expected_label=...)`核对了label与`_signature(atmo)`；
阶段03与阶段04当前runner按这条链逐高度建source，没有上述手工交叉绑定。

root现新增：`table.atmo is atmo`；真实PointSolarTable还以已有`_signature`核对
`table.definition`，不另造signature语义。独立复验结果：

- wrong-high/sea-table构造立即抛ValueError。
- 正确原始construction仍接受。
- 在原Atmosphere facade上通过公开aerosols setter换medium后，即使table.atmo指针相同，
  新构造也因definition不同被拒绝。
- 数学/源项测试18 passed / .29s。

证据：`independent-final-contract.json`。当前source SHA为
`6ed1b03c3c49ad748540064d90ce8306b8855f08d2409ba9a1cc51e1a654dea2`。
tracked `curved_transport_audit/source-binding-guard.patch`只新增这6行前置检查；
`runtime-source-provenance.json`保留原运行hash和新hash。它不改数值kernel、RNG、源表或
估计器。因此原本正确绑定的运行仍有效，不能仅因全文件hash变化要求物理重跑；
也不能忽略任意别的hash差异，须像此次一样读实际差异。

## 2. 归一、方向与n²组合

**有限盘。** 若S为盘心方向法平面的太阳辐照度，均匀日面radiance为
`S/[πsin²α]`；真立体角积分权重和应为`2/(1+cosα)`，法向通量
`sum(w cosρ)=1`。代码以球冠cosρ作Gauss径向积分，完整二维方位半步移位，
并不把权重和强制归一成1；遮挡节点以T=0保留，也不重新归一剩余可见盘。
此项与既有解析Rayleigh盘矩、horizon flux、opaque半盘测试相容。

**太阳源与detector。** table给出local
`G=(n_point/n_inf)² dΩ_app/dΩ_true`，source再乘`(n_obs/n_point)²`。
最终恰为`(n_obs/n_inf)² dΩ_app/dΩ_true`；local的n²既没遗漏也没双计。
无损传播使用reduced radiance `L/n²`，不在每段flight上额外乘折射率因子。
散射核两边处于同一local n；Lambert地面的continuation为rho，同样不再加n²。
这与此前不使用角Jacobian公式的Cartesian parallel-ray-tube能流验证是相容的，
没有仅以两个相同角导数公式互证。

**相位符号。** backward view和source-to-Sun方向都是真实光子方向的反向，
相位点积不变；scatter重构local apparent方向，ground使用正的local sin(app)。
连续flight rewrite把ATM NEE、continuation `_rotate`都换为arrival direction，
null collision的d_new也保留arrival；未看到把出发方向残留进相位核的路径。

**点源表。** direction、增益来自同一个插值式，vertical极限使用导数平方；
log1p species columns保持正值。shadow-aligned高度插值保住几何端点，但不保证任意
中间高度/角度均达到建表midpoint阈值；独立heldout误差是本次观测最大值，不是全域定理。
`_signature`明确是采样契约，不证明任意可变callable等价；当前显式frozen标准场景的限定必要。

阶段03现有报告对这些范围的表述适当；尤其没有把SS日面收敛冒充MC所有vertices的误差上界。

## 3. 连续flight：新的独立Hamiltonian链

本次低成本探针已固化为`tools/check_curved_hamiltonian.py`，未运行MC。
参考求解改用动量`p=n*d`：

```
dx/ds = p/n(x)
dp/ds = grad n(x)
```

它与flight的单位方向投影加速度不是同一状态/实现；使用DOP853、自适应误差控制和
独立sphere events，不调用其RK4、Hermite dense polynomial或径向转折判别。
折射廓线本身仍有意共享，检查的是传播实现而不是独立验证大气折射率模型。

在ground0/2.4两域，各取8条，包括1cm高掠地、5cm高掠而不触地、.2°低仰角出射、
38/100km负仰角穿过高层knot、下行触地、近竖直及短程trial，比较2/1km最大步长：

| 指标（本次32条） | 2km | 1km |
|---|---:|---:|
| ground/top/trial分类分歧 | 0 | 0 |
| 最大位置差 m | .022482 | .002272 |
| 最大累计弧长差 m | .021803 | .002203 |
| 最大到达方向向量差 | 1.98e−7 | 2.38e−9 |

参考解的`abs(|p|/n−1)`最大1.92e−10。
这支持当前2km实现的轨迹精度，不能自动变成天空radiance误差界；near-shadow顶点可能
把几何变化放大，最后仍需实际MC步长比较。完整数值与代码hash在
`independent-final-probes.json`，执行摘要在同名log。

复跑：

```bash
uv run python -m tools.check_curved_hamiltonian
```

## 4. MC组合中可证明的与继承的边界

全局majorant来自MC fine-grid的ext节点最大值乘1.005。实现的ext在节点间是正的
log-linear插值，故该majorant控制的是**这份离散光学场**。以真实弧长抽指数分布并沿折射
轨迹走，不要求再乘n或按高度重加权。它不保证任意连续窄峰都被fine-grid采到。

RNG/scattering/source calls没有进入RK substeps；source-only替换不改变路径的断言也有
归档的16组fingerprints支持。global majorant变更会改变null-collision的随机耦合与
方差，不能把固定seed输出差异直接解释为曲率，亦不能要求其逐位等于coarse-majorant结果。

已知继承边界应继续保留：

- total scattering和各component分别log插值，NEE与continuation之间存在微小不一致；
  前次composition review已测局部excess约1e−5至6e−5，不是radiance误差上界。
- aerosol continuation使用2049点uniform-mu inverse-CDF近似，而NEE调用adaptive phase。
  本次直接积分实际inverse-CDF每个cell的概率，和Mie解析Legendre矩比较：rural在
  450/550/650nm的β1误差约−1.36e−5/−1.41e−5/−1.37e−5，β0..4最大差3.14e−5。
  8193/32769点仍有约3e−5的phase-table近似底，不应把全差都归给2049采样网格。
  当前clear场景未见百分比级低阶矩问题；不推广至云/粗粒子，也不换算成MS误差预算。
- 50阶上限、RR尾部和各数值插值都继承自原MC；一次项闭合没有检验完整高阶尾部。
  仅有8个ensemble的SEM/Student区间不证明未知rare-path尾质量已收敛。

这些是实验范围，不是发现了一个已证实的20%高阶遗漏。

## 5. 独立核对新日面证据

审读期间主线补齐了tracked `curved_transport_audit/disk-convergence/`。
我独立核验其输入manifest全部SHA，并从raw谱重新积分photopic，不仅采信summary：

| 16→64盘面比较 | ΔMS或Δsplit cd/m² |
|---|---:|
| global-straight，0km，8 seeds | ΔMS=(1.21558±1.25340)e−5 |
| global-straight，2.4km，8 seeds | ΔMS=(.63675±1.17930)e−5 |
| curved，0km，seed310 | Δsplit=3.21614e−5（.006277%） |
| curved，2.4km，seed310 | Δsplit=3.30303e−5（.007633%） |

每对path fingerprint、NEE调用/点数、flight_stats、table SHA和source hashes均逐项相同。
这是实际采到vertices的独立重加权证据，明显强于仅以SS收敛代表MS；但curved只有一个seed，
不能覆盖rare vertices或所有场景。主曲率比较仍应使用disk16/16匹配source，勿混入disk64效应。

## 6. 可以最终签发的陈述

1. **阶段03源项实验成立。** 本次再次核验归档22份输入SHA；其48条全光谱的
   photopic结果、source配对、同一源的一次项统计相容性有可回读证据。
   0.493233/.459769及高/海.932154是明确场景、直线漫射、有限预算的实验估计。
   相对legacy的变化包含SS/source处方变化，不可全归单个G；legacy本身是混合源参照。
2. **气溶胶网格缺陷是独立、真实的介质定义问题。** ce550固定归一解决query-grid依赖；
   全81网格旧实验可复用不意味着内部anchor的twilight结果不变。原文已正确区分。
3. **参考边界和filter-only结果可以签发。** 未分离夜背景的总V、historical-night示例、
   photopic区别以及实际−6°约.5–.7%滤镜修正都有来源/数据；没有因此因果识别剩余绝对量。
   −12°的16.8%示例绝不能写成已实测扣除了同期夜背景。
4. **连续折射传播机制已获得额外独立支持。** 当前没有新增未处理的组合阻塞项；
   正确的局地方向、reduced-radiance权重、arc-length majorant和边界事件均可具体说明。
5. **阶段04全高阶数值继续待主线判定。** 核对同source table、盘面、λ、真实太阳角、
   observer/ground、光学场与path budget；明确8×10000 curved和8×100000 control的预算差。
   paired seed可估协方差，不意味着matched path；不要挑选或删除不利seed。
   目前被报告的coarse/global高站差仍需分析，本文不把它自动解释为majorant物理改变或
   自动归零为“只是噪声”。最终数值结论须列清残差、步长、采样和未识别范围。

三份tracked报告目前没有要求撤回的范围过度声明；本文给出的签发边界应继续进入阶段04收口。

## 永久归档与复跑

本审读固化于`validation/data/curved_transport_audit/final-review/`。
`independent-final-probes.json/.log`、`historical-initial-probes.py`保留最初运行版本，
没有改写其中guard前的source hash；`independent-final-contract.json`保留guard修复后的独立拒绝复验。
`source_provenance.json`与`inputs.sha256.json`明确区分初始记录、迁移后工具和当前portable重跑。
默认工具新结果写`hamiltonian-probes.json`；这不覆盖最初的JSON/log，`--out`可指定其他路径。
从仓库外使用`uv run --project <repo> python <repo>/tools/check_curved_hamiltonian.py --out <file>`。
这只检查flight与phase moments，不运行MC，也不改阶段03/04 ensemble或validation目标。
