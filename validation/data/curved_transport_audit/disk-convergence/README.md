# MC 太阳盘16→64点：冻结配对证据

范围：太阳−6°、天顶、clear、full81（380:5:780nm）、每seed/λ 10000paths。
16点为2×8方向，64点为4×16方向，同一均匀16′盘、连续太阳源和冻结源表。
NPZ引用 `validation/data/unified_solar_source_audit/`，不重复复制。

- `global-straight.jsonl` 与 `disk64-global-straight.jsonl`：两高度，各seed310–317。
  16/16组source point fingerprint、NEE调用/点数、flight_stats、table SHA及source实现SHA相同。
- `disk16-curved-ground*.jsonl`：从主线曲线结果按原行bytes截取的两个seed310控制记录；
  原文件仍在追加，本目录的控制已经冻结。
- `disk64-curved-ground*.jsonl`：各高度seed310，2km RK步长、64点日面；
  同高度16/64的point fingerprint、flight_stats、table/source SHA逐项相同。
- 三份 `.log`：64点MC运行记录；straight累计MC45.98s；curved海144.75s、高117.15s。
- `disk-convergence.json`：straight逐seed配对差、光谱重积分、SE/协方差与高度比。
- `curved-ground*-summary.json`：曲线固定seed的配对差；seed SE为null，因为每高度只有1seed。
- `recorded-library.patch`、`library-provenance.json`：straight两批head不同，但旧dirty patch
  正好等于两head间committed library diff，SHA256完全相符，证明运行时库内容一致。
- `inputs.sha256.json`：原始光谱、日志与provenance的11份输入哈希；派生summary不入循环校验。

straight配对ΔMS海(+1.2156±1.2534)e−5、高(+.6368±1.1793)e−5 cd/m²；
curved seed310 ΔMS海+3.9826e−5（+.0169%）、高+3.4101e−5（+.0249%）。
这是已访问vertices的聚合敏感性，不是任意单vertex或未采到rare tail的误差上界。
straight误差为8个独立seed的标准误；不把SS收敛直接当作MC顶点收敛证据。

从冻结记录重算，不跑MC：

```bash
uv run python -m tools.check_solar_disk_mc \
  --reference validation/data/curved_transport_audit/disk-convergence/global-straight.jsonl \
  --candidate validation/data/curved_transport_audit/disk-convergence/disk64-global-straight.jsonl \
  --out out/disk-straight-summary.json
uv run python -m tools.check_solar_disk_mc --curved \
  --reference validation/data/curved_transport_audit/disk-convergence/disk16-curved-ground0.jsonl \
  --candidate validation/data/curved_transport_audit/disk-convergence/disk64-curved-ground0.jsonl \
  --out out/disk-curved-ground0-summary.json
```

高站把最后一条命令的ground0改为ground2p4。复制本目录到另一位置，只需修改参数路径；
后处理不读scratchpad，也不依赖本机绝对路径。原raw记录的旧config路径保留为provenance。
source工具SHA是本冻结版本契约的一部分；若重跑MC使用后续source实现，应先审明代码差异，
不可单凭同波长/同seed将不同物理版本当作盘点消融。
