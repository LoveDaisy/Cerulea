# 连续曲率 MC flight：实现与有限预算验证

实现 `tools/continuous_curved_mc.py`。只新增本实验工具，未改生产库、统一太阳源或旧Snell工具。
旧Snell mid-index人工duct负结果仍保留在transport-design.md和artificial-duct.json。

## API与接线

```python
code, extras = rewrite_transport(code, atmo, max_step_m=2000, bend=True)
namespace.update(extras)
```

extras包含 `_layer_bounds`（只有ground/top两界）和 `_continuous_flight` 对象。
后者公开advance(x,d,s)，返回position/arrival direction/trial/ground/escaped，另可读取
calls/rk_steps/rk_ray_steps/turn_min_checks/ground_hits/top_escapes与last_distance。

- 一层global majorant由原tracer现有major构造取得全fine-table的βext最大值×原1.005；
  log-linear β表在任何区间都不超过节点最大值，所以这个delta tracking majorant成立。
- 原有一次 `rng.random(na)` exponential free path保留，RK内部不消费RNG。
- 替换旧直线flight与escaped计算；collision、analog absorption、species选择、phase sampling、
  RR、ground reflection和统计累积不重写。
- **所有 `da[ci]` / `da[si]` 改用arrival方向**，包括root新source.scatter的入射方向和_rotate基轴。
- 在L/n²变量下纯连续flight不额外改throughput；source自己负责nobs²/nvertex²。
- bend=False或全域常n时直接走解析straight，返回原direction位值，不靠RK逼近该极限。
- 字符串改写带严格锚点计数，拒绝叠加旧Snell或重复连续flight rewrite；不inspect已编译clone。

## 连续路径与边界事件

ODE使用与public RK相同n(r)定义但批量积分：dx/ds=u；du/ds=grad ln n-u(u·grad ln n)。
每个自由程分为≤max_step_m的RK4子步；end direction归一。边界使用起止x/u的三次Hermite路径，
先检查径向down→up的minimum，再在ground前半段与top各自括起根，选择最早事件。
连续单调nr的物理ray无径向maximum，故不能只看end radius的ground漏检已显式处理。
Hermite事件是可精化数值近似，不是对任意锐变profile的数学精度认证。

## 方向与Bouguer精化

独立public Cartesian RK参考ds=250m、止于同一modeltop。ground0/2.4，z38/z47的近切线
太阳方向以及各ground的.2°/30°向上方向，共8条ray；终点Bouguer与完整路径的选定检查均记录。

| 最大步长m | 8条ray最大方向差deg | 最大终点Bouguer相对差 |
|---:|---:|---:|
|4000|7.6902598e-05|7.7864934e-07|
|2000|1.4180619e-05|1.4332993e-07|
|1000|3.0772642e-06|3.1023598e-08|

某些单点角误差会有步长相位波动；上表为有限样本最大误差包络，不宣称全部空间全域精度。
另对z38近切线与ground.2°两条ray逐步测量Bouguer，最大沿路漂移：

- 4000m：7.4643558e-10
- 2000m：2.5668001e-10
- 1000m：2.8098857e-11

## 边界与零折射限制

- 全域n=1与1.000278（包括远场），curved与analytic straight的5个返回字段全部逐位相同。
- 2km一步的两端都在地外、但中间进入ground的case，turn minimum检验正确找出首个ground。
  起点高0.01m、app=-.005°：2km模式first hit126.54486385m；独立径向弧长积分126.54487869m，
  差约1.5e-5m。1km模式126.54486878m。没有把跨地之后的外侧终点当作collision。
- 同方向起点高0.05m的近切线miss仍为trial；径向向上距top .01m的ray先escape，而非跑完整freepath。
- 同global-majorant、bend=False clone对原straight tracer的512-path结果全部6字段逐位一致。
  另用日间sza80°确认非零信号下仍逐位一致；见straight-baseline-check.json。
- 同一小样本curved=True运行全部有限并有正信号，仅为接线/执行smoke，不能拿512条判定曲率亮度效应。

## 性能与限制

- 单次1000条free flight，平均抽样弧长50km、max step2km：0.0304s，25818 ray-substeps。
- 完整512-path MC的日间smoke：straight 0.021s，curved 0.372s，后者56988 ray-substeps。free-flight benchmark不能冒充完整MC开销。
- global majorant避免coarse-major边界歧义，但高层null collisions/离开200km大气所需RK步数会提高成本。
- 当前没有长MC统计或全谱生产精度认证；owner应比较4/2/1km和预算/种子，不能只凭ray probe签发最终亮度。
- n(r)节点与public RK共享，验证的是积分/事件机制，不是标准气氛是否匹配实测。

## 复跑

```bash
uv run python -m tools.continuous_curved_mc \
 --out scratchpad/explore-twilight-altitude/phase-04-curved-diffuse/continuous-probe.json
```

代码实现及probe已冻结，source接口和统计由owner接入；不修改任何已有物理基准。
