# Native coarse / global-straight majorant 等价性审计

2026-11-01；只读审计，未改任何库或tools实现，未追加MC运行。
复现脚本和数值证据为本目录的 `majorant-equivalence-probe.py/.json`。

**可签结论：两种方法以同一个fine-grid光学场、太阳源、物理边界和散射/终止规则为目标；
本批全81波长、两个高度均未发现majorant漏界或散射阶数错配。原方法有0.1mm穿层位移，
所以不宣称逐位或严格浮点等价。高站的2.24个合并标准误差异仍作为独立采样边界保留，
既不认定缺陷，也不签发统计已完全闭合。**

## 1. 光学与源是否相同

`continuous_curved_mc.rewrite_transport(..., bend=False)`仅替换flight block、escape block及
phase/rotate的方向变量；extras只提供 `_layer_bounds` 和 `_continuous_flight`，
**没有替换 `_fine_grid`**。该fine-grid函数仍调用原来的 `_layer_bounds_z`，所以旧coarse
边界节点继续保留。将transport变为global一层，不会顺带变更光学网格。

实际构造的网格为海平面761点、高站760点；均为原700个quadratic intervals，再并入
atmosphere bounds及coarse layer heights。脚本记录了fine-grid与beta_ext数组SHA256。
捕获source rewrite之后、transport rewrite前后的源码比较证明：

- 整个optical coefficient、log表和phase-table构造代码逐字相同。
- ground事件、collision接受、albedo、species选择、散射抽样、order cap、Russian roulette
  和commit，除 `da`→`arrival` 的预期重命名外逐字相同。
- bend=False走解析直线分支，arrival direction是原方向的精确copy；不是小步RK近似。

阶段三之后生产库只改了 `skycolor/aerosol.py`。本次另从596aaba取出原
`MieAerosol.relative_coefficients`方法，在同一材料states上独立重执行，确认full81包含
550nm时，每个aerosol的ext/sca数组与当前实现**逐位相同**，不是只接受已有修复报告。
气体密度、截面、fine-grid与MC实现未变。

阶段三源文件的SHA精确对应已提交的7ed71a62版本。对照当前AST，`solar_disk`、
`_geometry`、`scatter`、`ground`和`single_zenith`的数值实现全部相同；新增的是构造契约
和可选transport组合。两批各高度的NPZ SHA相同，所有seed的确定性SS数组也逐位相同。

## 2. Majorants是否真控制消光

native边界（km）：海 `[0,1,4,12,40,120,200]`；高
`[2.4,3.4,6.4,14.4,42.4,122.4,200]`。global只有各自ground/top两个端点。

native每层和global均用对应fine nodes的最大beta_ext乘1.005。MC接受概率取同一
fine-grid的log-linear beta_ext；正函数的log-linear插值在每段的最大值位于端点。
fine-grid包含每个coarse边界，因而每个插值段完整处于某一layer内，两个端点均被该层
majorant包含。这给出机制上的upper-bound证明；不依赖仅仅“稠密采样没看见异常”。

另外用实际 `mc._interp_log` 在每个fine interval取33个等距位置，并追加各coarse边界
的两侧10μm与边界本身，经过真实radius→height回算后，用原searchsorted选择layer。
全81波长总计约393万(height,wavelength)点：

| ground km | dense heights | max beta/major native | max beta/major global | 漏界数 |
|---|---:|---:|---:|---:|
| 0 | 24241 | .9950248756218923 | .9950248756218916 | 0 / 0 |
| 2.4 | 24301 | .9950248756218925 | .9950248756218917 | 0 / 0 |

直接连续密度构造的beta在本批dense点也未超过native majorant，但MC的严格保证针对
**实现所用插值场**；任意亚网格窄profile不受本标准clear场景的dense证据认证。
`p_real`的clip没有在本批场景掩盖任何majorant不足。

## 3. 边界、随机过程与有限epsilon

native在某层内抽 `s=-log(U)/M_layer`，若先到层界则进入相邻层并重新抽样。
指数分布的memorylessness保证这种边界重抽样合法；每点被接受的碰撞强度为
`M_layer*(beta/M_layer)=beta`。global换成常数M，仍有相同的thinning强度beta。
因此majorant影响null collision数量和随机数消耗，不改变理想离散光学场的目标分布。

两者ground/top均来自相同atmo的最内/最外radius：ground=observer=0或2.4km，top=200km。
native与global都用解析球面直线交点，先ground则反射、先top则退出。global的ground
discriminant允许等于0，native用严格大于0；恰好数学切线的差别为连续方向分布的零测集。
ground反射后的1μm径向抬升在两者相同的未改代码中。

**没有逐位等价的承诺。** native每次非ground的layer exit额外前进
`_EPS_MOVE=1e−4m`，global直接停在top/ground，没有内部层界跳跃。因此native会跳过
极短的内部消光段。以本批全波长最大beta估算，每个crossing跳过的消光深度尺度至多为
海1.011e−8、高5.130e−9。这是**单事件尺度**，不是未计路径数量/源梯度后的全局radiance
误差上界；未发现支持把它归因为高站约.00884cd/m²差额的证据。

## 4. 散射阶数和其他近似

两批driver都未覆盖 `mc_radiance_split` 的默认 `max_scatter=50`，且当前MC文件相对
阶段三未变。达到第50次大气散射时，该次NEE已累计，然后终止continuation；同一顺序
适用于native和global。null collisions及内部layer crossings不增加散射阶数。

两者同样采用RR_WEIGHT=.05、RR_START=10、RR_SURVIVE=.75；ground albedo、species
抽样、phase CDF、含地面路径归入multiple的split规则不变。200000步guard在两种方法
都属于报错出口，不是静默截断；global额外null events可能改变达到guard的成本，但本批
记录没有该错误，不能据此解释统计差异。

既有“total scattering与各species分别log插值”的微小不一致仍共同存在，见
`composition-review.md`；它不是本次两个majorant之间新增的差异。

## 5. 统计差额只能如实保留

原光谱独立重积分得到：

| ground km | phase3 native SS+MS | phase4 global100k SS+MS | global−native / combined SE |
|---|---:|---:|---:|
| 0 | .493233405±.003080821 | .492102224±.002616641 | −.001131181±.004042062；−.280 |
| 2.4 | .459769427±.003474424 | .450925106±.001859374 | −.008844322±.003940672；−2.244 |

两批分别seed200–207与310–317，集合不重合；合并SE用两个独立seed均值方差之和，
不能套source-only共享路径的paired SE。即使未来复用相同seed，majorant改变也会改变
null/边界随机数消耗，不能假定路径配对。

每批只有8个独立seed，经验SE与尾部仍有不确定度。现有mechanism检查没有找到
majorant漏界、换fine-grid或阶数错配；这支持继续把两者看作同一离散传输问题的
不同抽样实现（加上明确的有限epsilon边界差异）。它不证明2.244SE必由偶然抽样造成，
也不证明高站统计已经完全闭合；最终报告应保留这项残差，不将两个批次强合为一个精确值。

## 复跑

```bash
uv run python scratchpad/explore-twilight-altitude/phase-04-curved-diffuse/majorant-equivalence-probe.py
```

约1秒，只编译捕获source/transport克隆并评估密度与插值，不追踪任何新光子。
JSON保存实现SHA、fine-grid/beta SHA、全波长界限、常数和两批独立统计，供主线引用。

## 6. 新控制追加：原差额保留，另做同seed批次比较

原先跨seed200–207/310–317的高站差 **−2.244SE** 原样保留，不用新批替换。
主线随后追加native、seed310–317、每λ100000paths、ground2.4的8条记录，得到：

- 新native SS+MS = **.454110165522±.002181267932 cd/m²**。
- 同seed global SS+MS = .450925105795±.001859374187 cd/m²。
- 逐seed的native−global均值 = **.003185059727±.003656151461 cd/m²，+0.871151SE**。

这次SE直接来自8个独立seed批次的差额，因此允许同seed两实现之间存在任意实际
协方差。它是**seed层面的配对**；majorant改变后，null collision历史和光子轨迹不同，
没有宣称source-only消融那样的同路径配对。原先跨不同seed批次的2.244SE仍按独立均值
合并方差处理；两类SE不混用。

新控制不支持把原差额认定为已证实的majorant系统偏差；也不能证明所有系统偏差为零，
或尾部/SE已经完全收敛。所有旧/新原始记录均保留，没有按是否更接近某个预期筛选批次。

## 7. 永久探针、版本契约与主报告范围

上述审计已固化为 `tools/check_majorant_equivalence.py`。默认根目录来自`__file__`，
默认输入为tracked数据；基线Mie/MC/solar-source均有原源码snapshot，不再依赖git历史查询
或scratchpad路径。它校验17份majorant-control冻结输入及phase03既有22份冻结输入，
重建全81波长dense界限，再从原光谱重算历史对比与新控制，无新增光子。

```
uv run python -m tools.check_majorant_equivalence
```

永久归档：`validation/data/curved_transport_audit/majorant-control/`，包含本报告追加前的
历史快照、原probe脚本/JSON、六份来源snapshot、guard精确patch及新probe JSON。
更新后的完整报告另存同目录 `majorant-equivalence-audit.md`。

当前source文件SHA与早期global记录不同：runtime原值为5d25bf55…，新native/current
为6ed1b03c…。归档before/after源码逐字相减，恰好只有table construction identity和
既有signature的六行前置检查；所有solar disk、geometry、scatter、ground和SS数值AST
均相同。新探针核对每个raw record对应的source SHA、其余共同工具SHA、源表SHA及
确定性SS逐位相同。这里的“hash差异”对应已审明的guard-only变更，不能把任意未来
源码hash变化自动豁免为无数值影响。

只读审阅`docs/CURVED_TRANSPORT_AUDIT.md`中majorant段落：其保留原2.25 combined-SE、
追加0.87SE控制，并说“原先跨种子差没有转化为已证实的majorant系统偏差”，表述合范围；
没有宣称严格算法逐位等价、所有majorant偏差已排除或MC尾部已经收敛。本项无需修改主文。
