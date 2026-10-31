# ground2.4进程退出139：低成本只读审查

审查时间：2026-11-01 01:32–01:34 +08:00。
结论：**当前原因未知**。没有找到这次Python crash报告、native栈或确定的非法数组机制，
不能据此归因于Accelerate、多线程、内存不足、某个seed或曲线路径算法。未修改源码、
包版本、线程设置，也未重复长批。

## 已知运行事实与取证界限

主线报告：原ground2.4长批在seed312落盘后、seed313进行时退出139，没有Python
traceback。退出码139按常见shell约定对应128+SIGSEGV(11)，但原运行工具的完整状态由
主线保留；仅凭此次日志无法独立还原崩溃指令或C栈。

独立回读 `curved-ground2p4.jsonl`：当次审查时恰有三个完整JSON记录，seed310/311/312，
无半行JSON；原log也恰有这三个完成记录，没有异常栈。原log大小879字节，SHA256：
`ffdf7a95a78a6ff9f56ef21d3ef2fa52efebeb728ad4c3b0acc31214f85ef79a`。
JSONL当时大小56151字节，SHA256：
`504ae71bd10eafc8ab07aa02120baad80545a7c5da246a3014bd46205fdb4256`。
原进程seed313未产出结果，不能把它当作零值或纳入统计。

主线已用 `python -X faulthandler` 从seed313续跑；审查时进程PID67514存活、约99%CPU，
resumed log只有table_and_single记录，尚无seed313完成或Fatal Python error。
这证明续跑已开始，**尚不能证明同seed成功或问题不可复现**。海平面长批也在运行。

## 系统诊断文件与环境

- 扫描用户 `~/Library/Logs/DiagnosticReports`（含Retired，122个文件）和系统
  `/Library/Logs/DiagnosticReports`（32个文件）。没有此次Python/uv crash报告；
  系统最新Python文件为2026-10-30的`.diag`，与本次事件不对应，未拿它推断本次原因。
- `/usr/bin/log show --last 15m --style compact` 过滤ReportCrash且消息包含Python或
  sky-color，返回表头、无匹配记录。缺日志不构成“没有native fault”的证据。
- 此诊断shell `ulimit -c` 为0，`/cores`为空。没有core dump可供离线定位；原进程是否
  完全继承同一limit未单独取证。
- Python 3.12.11，macOS14.7 arm64；worktree `.venv` 的NumPy2.5.4、SciPy1.18.1。
  `np.show_config()`和`scipy.show_config()`均报告Apple Accelerate BLAS/LAPACK。
- `otool -L numpy._core._multiarray_umath...so` 直接确认Accelerate.framework链接。
  `scipy.interpolate._ppoly...so`只显示libSystem；`scipy.special._ufuncs...so`显示
  libc++/libSystem。故不能把“有SciPy调用”直接等同于“该调用进入BLAS”。
- 本诊断shell中 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`、
  `VECLIB_MAXIMUM_THREADS`、`NUMEXPR_NUM_THREADS`、`BLIS_NUM_THREADS`、`OMP_DYNAMIC`、
  `MKL_THREADING_LAYER`均未设置；launchctl的VECLIB变量也为空。机器12个物理/逻辑核。
  未安装threadpoolctl，未额外安装。**未设置线程变量不等于实际BLAS线程数为1或12**，
  也未读取原故障进程环境，不能以本shell设置替代其运行时线程状态。

## 数组与native边界审查

只读检查 `continuous_curved_mc.py`、`unified_solar_transport.py`、
`point_solar_table.py` 和aerosol phase调用链：

1. 未发现ctypes/cffi/自定义扩展指针、`as_strided`、外部buffer或自定义native并行代码。
   当前native边界主要是NumPy数组运算/`einsum`/矩阵乘法/插值，以及SciPy spline评估。
   存在native调用只是定位范围，不构成对其中任一库的归因。
2. RK起始position/direction显式copy；alive/事件数组为正常NumPy整数/布尔索引。
   Hermite事件的lookup仅用于已由candidate索引设置的event；本次白盒未发现把未分配
   native内存交给该路径的机制。即使数值越界，正常NumPy索引应抛Python异常，不能
   反过来断言底层库绝不可能存在bug。
3. source query在进入行插值前检查有限值、height/elevation域；行索引和cell索引clip。
   非有限位置或非法形状通常应变成ValueError/IndexError，而不是无栈退出。
   本次没有保存故障前的逐调用数组，无法验证最后一次实际输入是否满足全部条件。
4. NPZ加载后profile/rows等数组显式copy，不是关闭npz后遗留的mmap或外部指针。
   source的chunk评估及相函数表插值使用正常NumPy数组。未见use-after-free证据。
5. Python侧调用串行。并行运行的ground0/ground2.4/straight批次是**独立Python进程**，
   不是共享同一aerosol cache数组的Python线程。没有据此确认进程内的native线程数，
   但不能将跨进程并行直接当作共享NumPy数组竞争。

这次审查没有找到可据以修改算法或依赖的确定根因。先保留已落盘三seed及原log，
让已有faulthandler续跑提供下一条独立证据。如果再退出，可优先依据其Python/native栈
缩小到具体调用，再做对应小输入复现；不在没有栈的情况下试改物理算法或包版本。

## 审查结束时新增证据

主线已有续跑刚完成seed313（116.4167秒），未见faulthandler异常输出：
total=0.4699154379、single=0.2863792766、multiple=0.1835361613、split_total=0.4791600842。
因此原故障**未在该seed的这次新进程复跑中重现**；这不排除原长进程已有状态、资源或
外部条件参与，也不证明已消除根因。无需重复该seed或更换数值配置来获得同一证据。
