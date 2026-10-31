# 独立连续视线一次散射闭合

在 ground=0、太阳−6°、550 nm、方位0°的三条视线上，独立 DOP853 一次积分与
continuous RK Monte Carlo 的 `single` 在目前统计精度内一致。70°最初16个低样本种子
偏低11.46%，补充**不同种子**的高样本批次后偏差为−0.50%（−0.26个seed-SE）；
因此不能把前一批偏差解释为已确证的传输系统误差。

使用完整clear81大气和经过签名校验的ground0 refined太阳源表，只显式调用完整对象的
`_trace_wavelength(..., iw550, ...)`，没有缩减或绕过表的介质签名。表默认来自tracked
`validation/data/unified_solar_source_audit/tables/point-source-clear81-ground0-refined.npz`，
label=`phase3-clear-81-ground0-refined`；JSON保存其SHA256和相对路径。

## 独立链与数值检查

`tools/check_curved_single.py` 从 `RefractiveProfile` 独立构造n(r) CubicSpline，以
`solve_ivp(method="DOP853")` 解弧长参数的x、d和τ：

```
dx/ds = d
dd/ds = ∇log(n) − d(d·∇log(n))
dτ/ds = beta_ext
L_single = F_sun/(4π) ∫ exp(−τ) beta_sca source.scatter(x,d) ds
```

不调用 `ContinuousCurvedFlight` 来得到积分视线；固定弧长分段上各用8点Gauss积分。
2km与1km同时约束DOP853最大步长和积分分段。两者一次辐亮度相差
17.17 / 4.53 / 1.36 ppm（view0/70/85），Bouguer不变量漂移≤1.62e−10。

闭合目标明确选用MC的同一fine-grid log-linear光学场，避免把介质离散误差混入传输
检查。15个代表高度上，该场与直接连续密度算出的消光最大相对差2.93e−4，散射3.25e−5；
550 nm的完整光谱与 `_restrict` 后aerosol系数逐位一致。没有修改任何介质或core算法。

在每条视线选7个受照点，独立重建局部太阳单位向量，再计算
`mu_phase = backward_view · local_sun`。实际光子传播的入射/出射方向都是这两向量的
负向，因此点积符号保持；重构同时显式乘 `gain_local * n_observer²/n_point²`。
独立矢量结果与 `source.scatter` 相对差≤2.9e−14，检查了相函数方向和L/n²转移。

`max_scatter=1` 的single仅累计**第一次大气碰撞且此前没有ground反射**的NEE。
地面直接贡献归multiple，ground后首次大气散射也归multiple。此处三条向上视线在第一次
碰撞前不触地；后续运行另直接断言 `total==single` 且 `multiple==0`，实际全部通过。

## 统计结果

所有辐亮度单位为 W m⁻² sr⁻¹ nm⁻¹。先各视线5000条、16个独立种子，2/1km同种子配对。
下表采用2km；SE来自16个种子均值间的样本标准差除以√16，path-SE也完整保留在JSON。

| 视天顶角 | DOP853 | MC均值 | seed-SE | 相对差 | 差/seed-SE |
|---|---:|---:|---:|---:|---:|
| 0° | 3.64874e−6 | 3.84832e−6 | 3.96935e−7 | +5.47% | +0.50 |
| 70° | 4.85298e−5 | 4.29663e−5 | 1.74316e−6 | −11.46% | −3.19 |
| 85° | 2.34069e−4 | 2.31204e−4 | 6.25023e−6 | −1.22% | −0.46 |

70°再使用seed7400..7407、**各50000条**、只跑2km，共400000条额外路径：

- MC=4.82872e−5，deterministic=4.85298e−5，差−0.500%。
- seed-SE=9.37797e−7（相对1.93%），path-SE=1.12972e−6（相对2.33%）。
- 差/seed-SE=−0.259，差/path-SE=−0.215。
- 按路径数合并70°全部480000条：MC=4.74004e−5，差−2.327%，path-SE=1.01225e−6，
  差/path-SE=−1.116。没有丢弃或替换原低样本偏低的证据。

2/1km配对通常极近，但seed7312的85°single差2.62%，使16种子该视线聚合均值变化
0.158%。固定种子不保证浮点判别改变后逐条路径完全相同；没有把这条反例静默删除，
其原始记录保留。聚合步长变化仍小于当前2.7%的seed-SE，且独立确定性积分已到ppm级。

这些证据支持当前连续视线、统一太阳源的一次项闭合，尚不能证明全部波长、其他地面
高度或多次散射项的正确性；后者属于主线的独立ensemble审计。

## 复跑与文件

三个原始批次为 `first8.json` / `second8.json` / `high70.json`，同名log保存运行输出。
`independent-single-validation.json` 汇总逐种子数据、两类SE、合并估计、介质探针、相位/
n²探针、源表和代码摘要。三批实验计算耗时合计176秒；没有重复主线的full81全order实验。

从仓库任意工作目录可用工具绝对路径；以下按仓库根目录运行：

```bash
uv run python tools/check_curved_single.py --seed-start 7300 --out first8.json
uv run python tools/check_curved_single.py --seed-start 7308 --out second8.json
uv run python tools/check_curved_single.py --seed-start 7400 --seeds 8 \
  --photons-per-view 50000 --views 70 --steps 2000 --out high70.json
```

`--table` / `--table-label` 可显式选择同一完整介质的其他保存表。默认表位置由工具
`__file__` 定位仓库根，不依赖scratchpad或当前cwd；保留原签名检查。最后从`/tmp`以
`uv run --project <repo> python <repo>/tools/check_curved_single.py`执行2个种子、
各32条、70°的小样本检查，直接脚本入口和tracked默认表通过，证据为`portable-smoke.*`；
它仅验证入口，不参与上述
统计推断。源码未commit，core/golden未修改。
