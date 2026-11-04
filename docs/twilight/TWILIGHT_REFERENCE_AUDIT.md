# 暮光参考量与 V 通带审计

2026-11-01。此文固定原始观测定义、对照边界，以及对阶段03已有谱的低成本后处理。
本阶段没有新跑 Monte Carlo，没有调整太阳高度或拟合零点，也未修改 validation 结果。

后续2026-11-04的[原始观测核查](MATCHED_OBSERVATIONS.md)已取得FORS实际时间/方向/站高、
旧QE和同次暮光候选，并定位另一公开光谱数据的站点星历错配。
这里的“仍未知”保留本阶段的证据边界；新候选尚未完成完整响应、大气和绝对提取。

**−6° 的绝对量与高度差仍不能由这两组观测作因果识别。** Patat 的 V 观测包含未分离夜背景，
与模型 photopic luminance 也不是同一积分；Koomen 虽以 photopic 为仪器目标，高低站仍混入
季节、透明度和样本差异。实际计算的 ESO BESS_V 滤镜近似在本次−6°谱上仅产生约0.5–0.7%
总量修正，不能解释10–25%的差额。尚未知的完整仪器响应或太阳角度契约不能被虚构成校准误差预算。

## 原文事实

以下页码均为正式刊物；arXiv排版页码不同。来源及下载hash在
[`source_provenance.json`](../../validation/data/twilight_reference_audit/source_provenance.json)。

### Patat et al. 2006

[原文：A&A 455,385–393](https://www.aanda.org/articles/aa/pdf/2006/31/aa4992-06.pdf)，
[arXiv HTML](https://arxiv.org/html/astro-ph/0604128)。

| 位置 | 核对到的事实 |
|---|---|
| p387 §3 | 像素背景mode经零点转Johnson-Cousins表面亮度；未作colour/airmass correction；internal photometric error <1%。 |
| pp387–389 §4 / Table1 | 样本在天顶40°内，拟合95≤ζ≤105°；V系数(11.84,1.518,−.057)，RMS=.18mag。 |
| pp388–389 §4；p393 A.5 | 观测中夜背景逐渐显现；理论曲线另加Patat2003 Table4夜背景；未描述观测夜底扣除。 |
| pp389–390 §4 / Fig10 | CrAO600m，2002-12-09至11三次清晨，Tycho2定标，V色修正约.01mag；与Paranal天顶20°内、无月、太阳低5.5–11°样本比得ΔV=.27±.03mag，压力743/961hPa。 |
| p391 §5 | 分子密度、低层气溶胶或两者的贡献尚未区分。 |
| p387 §4；p388；p391 A.1–2 | 计算太阳高度；给出时角/赤纬关系；附录忽略折射、使用点太阳；观测轴未明确真/视及盘心/上沿。 |

仪器bias扣除和 astronomical night sky 扣除是不同操作。Sky mode可压低可分辨星点污染，
但没有把弥散airglow、zodiacal light从像素背景里分离出来；不能把Table1叫作纯太阳散射。
论文内部精度和样本RMS亦不能改名为外部绝对校准误差。

### 夜背景溯源

[Patat2003，A&A 400,1183–1198](https://www.aanda.org/articles/aa/pdf/2003/12/aa3223.pdf)，
p1190 §5/Table4：历史V夜背景均值21.61 mag arcsec⁻²、样本RMS=.20mag；该表的V值仍含
zodiacal bias，另列若补偿该贡献应施加的.18mag修正。这不是纯airglow，
也不是已扣除zodiacal light的常数夜底；2000–2001样本不是2006论文全部暮光图像的同期背景。

### Koomen et al. 1952

[原扫描：JOSA 42,353–356](https://static1.squarespace.com/static/54694fa6e4b0eaec4530f99d/t/5e876886cdfbaa441f538327/1585932427823/Measurements%2Bof%2Bthe%2BBrightness%2Bof%2Bthe%2BTwilight%2BSky%2B1952.pdf)。

| 位置 | 核对到的事实 |
|---|---|
| p353 仪器段 | 绿滤镜光电倍增管，目标为明视觉眼响应，1.5°视场；抵消暗电流；以Macbeth仪比照日间天空定标，并作钨灯/镭源传递检查。 |
| p353右栏 | 低于.003 candle/ft²，作者提示晚暮光photometric correction未施加，未给误差预算。 |
| p353末段 | 高站2800m、1951年5–6月七个无云无月晚间；低站30m、1–3月；垂直太阳透过率85–90% / 75–85%。 |
| p354 TableI/II；p355 Fig1 | 表值来自观测平滑曲线；−6°天顶高/低=.022/.060 candle/ft²，即.236806/.645834 cd/m²。 |
| p354表后；全文 | H为相对水平面的太阳高度；未明确盘心/上沿、真/视；未描述天文夜底扣除。 |

低亮度下物理photopic量仍有定义，不应因为视觉进入mesopic就把模型改成scotopic。
作者晚暮光校正段也不能为−6°制造某个误差百分比：其−6°天顶值高于文中阈值。

## 事实能支持的范围

### 夜底对深暮光的影响

同一V域中，使用上述**历史均值作示例**时，零点常数消去：

`f_dark / f_fit = 10**[−0.4×(21.61−m_fit)]`。

| 太阳高度 | 总V拟合 mag/arcsec² | 仓库近似V等效 cd/m² | 示例夜底占总V |
|---:|---:|---:|---:|
| −6° | 13.301 | .516445 | .0475% |
| −9° | 17.000 | .0171168 | 1.432% |
| −12° | 19.673 | .00145957 | 16.796% |

若示例夜底在线性V信号中扣除，−12° V等效目标约.00121442。
**这不是同期实测背景扣除，也不是推荐修改validation目标。** 它只说明−12°总观测与纯散射
的语义差异可能显著；在−6°，该历史夜底太小，不能解释20–30%差额。
计算值保存在[`reference-calculations.json`](../../validation/data/twilight_reference_audit/reference-calculations.json)。

### 站点差不等于受控高度实验

`.27mag → high/low=10**(−.4×.27)=.779830`，即高站暗22.0%；压力比`743/961=.773153`。
论文概述的“about30%”不能替代精确0.27mag换算。
两站都测总V背景，但未共用夜底、仪器、天气、气溶胶剖面或天空取样；Table1与站点比较的
天顶筛选也分别为40°和20°。因此不能把该比值作为同介质截断的高度导数。
Koomen高低比.3667同样混入季节/透明度；不能把不同论文的比值平均成“高度真值”。

阶段03统一源−6°海平面photopic约.4932，与Koomen低站.6458的比约.764，
属于跨场景数值残差；它没有识别尚缺多少某一种散射。
与Patat高站V等效.5164接近亦不是绝对正确性证据：高度、通带、背景、大气和取样未匹配。

### 太阳角度契约仍未知

两篇原文都不足以锁死观测轴的真/视及盘心/上沿约定。Patat的ephemeris、球面天文式和
无折射点源附录支持“几何太阳方向”的合理推断，不能证明观测数据计算程序的具体实现。
附录模型也不能补出观测盘缘定义；Koomen更未说明ephemeris方法。

仅作灵敏度演示：V拟合在−6/−9/−12°的局部斜率是1.404/1.062/.720 mag/deg。
一阶估计中，20–30%亮度差对应约.141–.203° / .186–.268° / .275–.396°位移；
假设差16′，局部亮度比约1.412 / 1.298 / 1.193。
这些不是角度误差预算，不能据拟合效果挑选偏移；进一步裁定需原始时间戳、FITS元数据、
ephemeris实现或作者确认。

## 对已有谱的滤镜实验

模型photopic量为`Y=683∫Lλ V_CIE(λ)dλ`，与标准V积分不同。
`108000×10**(−.4mV)`应叫V等效luminance，对任意暮光谱不是Y的恒等式。

本实验复用阶段03归档中48条−6°谱，六组（3种太阳源×0/2.4km）、各8 seeds，
λ=380,385,…,780nm。主量沿用原`split_total=deterministic_single+multiple`；
没有用legacy直接MC的single替代其确定性SS，直接MC字段也保留在输出中。

[ESO滤镜页面](https://www.hq.eso.org/sci/facilities/paranal/instruments/fors/inst/Filters/curves.html)
及其[BESS_V ASCII](https://www.hq.eso.org/sci/facilities/paranal/instruments/fors/inst/Filters/M_BESS_V.txt)
只给滤镜透过率。本实验假设谱段内QE=1、域外为零，未补CCD/光学系统响应；
不能把2007 HIGH系列套给2005旧TK2048数据，也没有将滤镜1200nm透射当作硅CCD响应。

photon约定信号为`Sγ=∫λLλT dλ`，`hc`在信号/零点比中消去；
另存`SE=∫LλT dλ`作为energy-detector诊断。AB零点固定3631Jy
（[SDSS官方定义](https://www.sdss4.org/dr14/algorithms/fluxcal/)），
`fλ,0=3.631e−23*c_nm/λ_nm²`，辐亮度乘`[π/(180×3600)]² sr`得到1arcsec²通量。

有条件的V近似另用固定[CALSPEC011 Vega谱](https://ssb.stsci.edu/cdbs/calspec/alpha_lyr_stis_011.fits)，
FLAM→W m⁻² nm⁻¹系数.01，固定Vega V=.031
（[STScI标准表](https://etc.stsci.edu/etcstatic/users_guide/1_ref_2_spectral_distribution.html)；该表当前用012，
本实验明确固定011）。photon下算得`mAB−mV,approx=−.0183475mag`。
没有依Patat或模型残差决定零点；这也不是重建原论文的Landolt系统变换。

### 总量与配对结果

±为8 seeds的均值SEM，不含系统或模型误差。C为均值之比。

| 高度/模式 | photopic Y cd/m² | photon V等效 cd/m² | C=V等效/Y |
|---|---:|---:|---:|
| 0km legacy | .563840±.003022 | .561437±.003344 | .995739 |
| 0km geometric | .431535±.003032 | .430588±.003344 | .997804 |
| 0km unified | .493233±.003081 | .490968±.003351 | .995408±.001991 |
| 2.4km legacy | .543492±.003464 | .540360±.004112 | .994236 |
| 2.4km geometric | .401072±.003469 | .399388±.004112 | .995801 |
| 2.4km unified | .459769±.003474 | .456781±.004083 | .993499±.002748 |

| 比较 | photopic比 | photon V比 |
|---|---:|---:|
| unified高/低 | .932154±.007479 | .930366±.006780 |
| legacy高/低 | .963913±.006184 | .962459±.005638 |
| geometric高/低 | .929406±.008024 | .927540±.007337 |
| unified/legacy，0km | .874776±.000852 | .874485±.000842 |
| unified/legacy，2.4km | .845954±.001085 | .845327±.001196 |
| unified/geometric，0km | 1.142973±.001021 | 1.140228±.001181 |
| unified/geometric，2.4km | 1.146353±.001383 | 1.143702±.001651 |

配对误差用同seed的`a−R b`残差估计，保留协方差。同高度source的path fingerprint相同；
高低站虽复用seed但路径不同，不能称相同路径重加权。

色谱依赖亦未被总量掩盖：unified SS低/高站C=.993223/.993646；
MS的C=.998205±.004534 / .993234±.007698。
MS的CIE(x,y)约(.20724,.19776)/(.20279,.19043)，比SS的
(.25449,.25789)/(.25018,.25271)更蓝，但本场景滤镜修正仍小。
若改用energy约定，unified总C约1.0066/1.0044，约定差约1%，不能制造10–25%解释。

所以已测得的滤镜效应保留source与高度结论。它没有完成旧FORS1总响应的复现，
也不能推广到−9/−12°或airglow线谱；后续精确合成V应补当年CCD×光学响应及标准星校准契约。

## 可复现产物

[`validation/data/twilight_reference_audit/`](../../validation/data/twilight_reference_audit)
包含原始滤镜、288KB的固定CALSPEC谱、URL/SHA、参考算术、逐seed/配对JSON和汇总CSV。
默认输入来自[`unified_solar_source_audit/raw/`](../../validation/data/unified_solar_source_audit/raw)，
不依赖scratchpad；脚本通过`__file__`定位仓库，输出来源使用仓库相对路径。

```bash
uv run python -m tools.summarize_bandpass_audit
```

脚本默认核对冻结输入SHA，随后重算：photopic相对原JSON闭合≤4.45e−16；
独立频率域`∫fνT/ν`与λ域AB差4.59e−11mag；平Fν重采样后AB误差4.54e−5mag；
光谱加倍的星等变化误差1.12e−16mag。复跑只需已有依赖，不下载数据、不运行MC。
