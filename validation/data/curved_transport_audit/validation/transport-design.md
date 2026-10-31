# Thin Snell transport：实现与负结果

实现 tools/curved_mc_transport.py，未改库或统一太阳source。
rewrite_transport只加薄层界面事件，NEE/scattering/RNG/throughput不改；majorant包含全部thin界面密度节点。
constantn或bendFalse不改变原direction/position；同thin grid的512光子MC逐位一致；Snell invariant/TIR解析检查通过。

独立点probe见probe.json：规则5°/30°向上射线呈一阶角误差收敛；地面5°为.004434/.002229/.001118°，
30°为.000657/.000330/.000165°（.5/.25/.125km）。近切线down ray却有非单调grid-phase误差：
z38、true -6附近为-.00962/+ .03346/-.07265°；细到.015625km仍约+.01088°。不能签发.125km精度充分。

更根本的限制：分层mid n虽逼近连续n，但每个向外n下降jump使q=n*r离散地向下跳，可形成连续单调q中不存在的
TIR trapping cavity。artificial-duct.json给出了a位于界面两侧q值之间、几何perigee仍在层内的解析构造；ray在
perigee与outer TIR界面间反复走，直到实际散射。该角域随layer减小，但暮光稀有近水平路径可能敏感。

因此按owner裁定保留为负结果，不扩大Snell MC统计投入。下一步改用连续RK轨迹+global-majorant delta tracking，
避免人工n jumps及曲线跨coarse-major边界歧义。source始终使用独立连续场定义。
