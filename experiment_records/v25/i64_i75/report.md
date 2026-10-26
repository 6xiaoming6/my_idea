# 20261025_v25_I64-I75最后一轮seed7输入优化_复核分析报告

> Git阅读副本：结果与分析保持原报告内容；图和小型审计表随附。完整密集诊断、原run及冻结源码仅在原服务器，评价目录为明确标记的压缩指标。

冻结套件：`/home/students/HuangMingYu/code/py/my_idea/my_idea/outputs/v25/experiments/final_input/a14c08aae45f4052`。十二次独立训练，全部seed7；I37/I56/I57为只读参考。自动报告形成于2026-10-24，本次于2026-10-25复核。原自动报告和冻结训练文件保持原状。

## 1. 实验具体做法与复现配置

|组|直接对照|实际做法|所检验问题|support/position/预测gain|
|---|---|---|---|---|
|I64|I37|只去掉逐轮专家输入的support|原骨干是否存在重复缺失结构特征干扰|0.0/1.0/1.0|
|I65|I37|只去掉逐轮专家输入的位置特征|重复位置输入是否不利于未见缺失形态|1.0/0.0/1.0|
|I66|I37|同时去掉逐轮support和位置输入|I57收益是否无需先清理预测补位就能获得|0.0/0.0/1.0|
|I67|I57|在I57上只恢复support，位置仍不重复输入|单独恢复缺失结构统计能否修复ID代价|1.0/0.0/0.0|
|I68|I57|在I57上只恢复位置，support仍不重复输入|单独恢复几何信息能否修复ID代价|0.0/1.0/0.0|
|I69|I66|I66再将专家缺失槽整块清零，路由仍读原统计|metadata关闭时是否还需要清理专家补位|0.0/0.0/0.0|
|I70|I66|I66只清理路由显式值统计，保留原专家补位|metadata关闭时是否只清理路由就足够|0.0/0.0/1.0|
|I71|I57|只恢复可见F在M/C缺失槽的聚合贡献，预测贡献仍为0|上轮删补位的代价是否含误删已知细尺度证据|0.0/0.0/0.0|
|I72|I71|保留完整可见F贡献，再加入0.25倍初始预测贡献|弱预测先验能否提高ID且控制OOD代价|0.0/0.0/0.25|
|I73|I71|保留完整可见F贡献，再加入0.5倍初始预测贡献|中等预测先验是否优于清零或原强度|0.0/0.0/0.5|
|I74|I71|I71加回0.25倍support和位置特征|弱辅助特征是否比全开/全关更好|0.25/0.25/0.0|
|I75|I73/I74|预定组合：完整可见F＋0.5倍预测＋0.25倍辅助特征|预测先验与辅助信息的弱注入能否互补|0.25/0.25/0.5|

### 公共模型、训练与输入定义

TaxiBJ沿用2452/342/707窗口；双通道12时间步，F/M/C空间32/16/8，三尺度真实观测各自缺失，训练为四类单缺失族混合0.4。H0三尺度编码融合、四轮共享八专家，每轮原生Top2；前三轮自由尺度、第四轮F；原受控差分、历史池、学习分配投影保持I37。

100epoch、batch32、每5轮ID验证MAE选best，GPU1单卡AMP串行；AdamW1e-3余弦至3e-4，wd1e-4，grad_clip_norm1。前后缺失F损失各0.5，可见rawF0.1，可见M/C一致性0.1，候选均衡0.01。均从相同未训练初始化独立训练，未加载已训练参考权重。

support_gain和position_gain仅缩放逐轮专家输入，H0和路由元数据保持；native mask始终不变。旧整块zero补位会同时删除缺失M/C槽中可见F的聚合贡献，因此不能简单称为仅移除预测。I71–I75先构造where(m_F,X_F,g×P0)，再进行原空间mean聚合；原生可见M/C优先输入其真实观测密度。仅P0缺失槽被缩放，可见F始终保留，派生值不成为新观测。P0保持初始预测、不detach。

手算例：2×2区域可见F为10、40，缺失P0为8、12，则g=0/0.25/0.5/1时fallback为12.5/13.75/15/17.5；g=1与原补位精确一致。I71(g=0)与I57整块zero有本质区别。路由observed选项仅修改显式值统计为可见F均值、预测统计置零；隐藏状态中仍含模型估计，不能说路由只接触观测。

ID含随机点、节点停测、时间间隙和空间区域四种单缺失族；二元组合含节点＋空间、时间＋空间、节点＋时间；未见形态含移动区域和多块区域；三元组合为节点＋时间＋空间。二元及形态各有两次mask评价。原六套测试协议保持；开发mask基种子2026102301–06，验证窗口；新确认mask2026102311–16，测试窗口，保留原split偏移。二元/形态各平均两次mask，再与三元类别等权；新mask不是新数据集或新训练种子。既有测试结果参与过方案设计，因此不是完全盲测。

先完成十二组开发评估并冻结selection.json，再开放本批测试。选择仅使用验证协议：ID≤I37同协议103%，按三类等权相对OOD MAE比值排序，同分依次OOD RMSE、ID、参数、编号。无合格组不强选新模型，不新增训练。最终推荐另要求OOD至少改善3%、最差类别退步≤5%。

原I37/I56/I57分别作为原模型、清理预测路径、再清理metadata的参考。I64–I66与I37构成support×position对照；I67/I68与I56/I57构成清理预测路径后的同类对照；I66/I69/I70/I57构成补位×路由对照；I71/I73/I74/I75构成预测强度×metadata强度对照，I72补充0.25预测剂量。

冻结源码/设计/配置和数据摘要见[plan.json](plan.json)，公共初始化核验见[initialization_audit.json](audits/initialization_audit.json)，CPU/GPU记录见acceptance（原服务器文件，路径见冻结plan）。
恢复入口：`python /home/students/HuangMingYu/code/py/my_idea/my_idea/outputs/v25/experiments/final_input/a14c08aae45f4052/source_snapshot/scripts/v25/run_final_input.py --suite /home/students/HuangMingYu/code/py/my_idea/my_idea/outputs/v25/experiments/final_input/a14c08aae45f4052`。

|组|冻结作业|原始run|
|---|---|---|
|I64|[配置](audits/taxibj_I64_seed7.json)|日志/检查点（原服务器文件，路径见冻结plan）|
|I65|[配置](audits/taxibj_I65_seed7.json)|日志/检查点（原服务器文件，路径见冻结plan）|
|I66|[配置](audits/taxibj_I66_seed7.json)|日志/检查点（原服务器文件，路径见冻结plan）|
|I67|[配置](audits/taxibj_I67_seed7.json)|日志/检查点（原服务器文件，路径见冻结plan）|
|I68|[配置](audits/taxibj_I68_seed7.json)|日志/检查点（原服务器文件，路径见冻结plan）|
|I69|[配置](audits/taxibj_I69_seed7.json)|日志/检查点（原服务器文件，路径见冻结plan）|
|I70|[配置](audits/taxibj_I70_seed7.json)|日志/检查点（原服务器文件，路径见冻结plan）|
|I71|[配置](audits/taxibj_I71_seed7.json)|日志/检查点（原服务器文件，路径见冻结plan）|
|I72|[配置](audits/taxibj_I72_seed7.json)|日志/检查点（原服务器文件，路径见冻结plan）|
|I73|[配置](audits/taxibj_I73_seed7.json)|日志/检查点（原服务器文件，路径见冻结plan）|
|I74|[配置](audits/taxibj_I74_seed7.json)|日志/检查点（原服务器文件，路径见冻结plan）|
|I75|[配置](audits/taxibj_I75_seed7.json)|日志/检查点（原服务器文件，路径见冻结plan）|

## 2. 完整结果、训练曲线与诊断

|模型|ID MAE|ID RMSE|二元MAE|形态MAE|三元MAE|三类OOD均值|ID变化%|OOD类别比变化%|最差类变化%|达标|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|taxibj_I37_seed7|8.051042|14.660256|8.297262|10.833551|7.892074|9.007629|+0.000|+0.000|+0.000|参考|
|taxibj_I56_seed7|10.296717|18.050757|16.239554|18.119822|16.292943|16.884106|+27.893|+89.808|+106.447|否|
|taxibj_I57_seed7|8.399249|15.008330|8.991531|12.961215|8.532077|10.161608|+4.325|+12.039|+19.640|否|
|taxibj_I64_seed7|8.082592|14.657872|19.844104|22.197179|18.257918|20.099734|+0.392|+125.134|+139.164|否|
|taxibj_I65_seed7|7.885394|14.068412|11.701053|14.835310|10.703403|12.413255|-2.057|+37.861|+41.023|否|
|taxibj_I66_seed7|8.153697|14.647412|18.015166|19.624659|16.494891|18.044905|+1.275|+102.425|+117.122|否|
|taxibj_I67_seed7|8.526790|15.055947|19.200657|20.777183|20.660745|20.212862|+5.909|+128.329|+161.791|否|
|taxibj_I68_seed7|9.013286|16.145297|20.025826|21.364098|25.637663|22.342529|+11.952|+154.470|+224.853|否|
|taxibj_I69_seed7|8.432462|14.830133|27.006370|24.167721|37.657826|29.610639|+4.738|+241.909|+377.160|否|
|taxibj_I70_seed7|11.278834|20.035906|24.021871|20.205986|18.054399|20.760752|+40.092|+134.932|+189.516|否|
|taxibj_I71_seed7|8.616543|15.237946|21.662547|19.875739|21.048852|20.862379|+7.024|+137.085|+166.709|否|
|taxibj_I72_seed7|8.705907|15.512609|16.828033|20.861507|16.058720|17.916087|+8.134|+99.619|+103.479|否|
|taxibj_I73_seed7|8.514158|15.030622|14.177228|18.009921|13.263602|15.150250|+5.752|+68.390|+70.866|否|
|taxibj_I74_seed7|8.920183|15.826327|12.654929|15.378203|11.500836|13.177989|+10.795|+46.732|+52.519|否|
|taxibj_I75_seed7|10.984407|19.770276|23.978550|23.481615|22.845008|23.435058|+36.435|+165.070|+189.468|否|

变化均相对同seed的原I37。OOD均值为三类别MAE平均，相对变化为三类别比值平均，两者分母不同。

### 原六套完整测试

|模型|协议|MAE|RMSE|
|---|---|---:|---:|
|taxibj_I37_seed7|in_distribution|8.051042|14.660256|
|taxibj_I37_seed7|unseen_combinations|8.310507|14.753401|
|taxibj_I37_seed7|unseen_geometry|10.821157|18.487384|
|taxibj_I37_seed7|unseen_triple|7.892074|13.917090|
|taxibj_I37_seed7|unseen_combinations_repeat|8.284017|14.577803|
|taxibj_I37_seed7|unseen_geometry_repeat|10.845945|18.493770|
|taxibj_I56_seed7|in_distribution|10.296717|18.050757|
|taxibj_I56_seed7|unseen_combinations|16.248368|27.974420|
|taxibj_I56_seed7|unseen_geometry|18.079501|32.745474|
|taxibj_I56_seed7|unseen_triple|16.292943|27.660796|
|taxibj_I56_seed7|unseen_combinations_repeat|16.230739|28.058184|
|taxibj_I56_seed7|unseen_geometry_repeat|18.160143|32.874721|
|taxibj_I57_seed7|in_distribution|8.399249|15.008330|
|taxibj_I57_seed7|unseen_combinations|8.978732|15.803427|
|taxibj_I57_seed7|unseen_geometry|12.994492|24.434769|
|taxibj_I57_seed7|unseen_triple|8.532077|14.967278|
|taxibj_I57_seed7|unseen_combinations_repeat|9.004330|15.836100|
|taxibj_I57_seed7|unseen_geometry_repeat|12.927939|24.031626|
|taxibj_I64_seed7|in_distribution|8.082592|14.657872|
|taxibj_I64_seed7|unseen_combinations|19.994881|41.297442|
|taxibj_I64_seed7|unseen_geometry|22.042078|45.382029|
|taxibj_I64_seed7|unseen_triple|18.257918|39.569013|
|taxibj_I64_seed7|unseen_combinations_repeat|19.693328|40.891142|
|taxibj_I64_seed7|unseen_geometry_repeat|22.352280|46.257332|
|taxibj_I65_seed7|in_distribution|7.885394|14.068412|
|taxibj_I65_seed7|unseen_combinations|11.823125|23.761234|
|taxibj_I65_seed7|unseen_geometry|14.849531|27.920917|
|taxibj_I65_seed7|unseen_triple|10.703403|21.597994|
|taxibj_I65_seed7|unseen_combinations_repeat|11.578982|23.152625|
|taxibj_I65_seed7|unseen_geometry_repeat|14.821089|27.610584|
|taxibj_I66_seed7|in_distribution|8.153697|14.647412|
|taxibj_I66_seed7|unseen_combinations|18.095956|36.306245|
|taxibj_I66_seed7|unseen_geometry|19.612206|39.417631|
|taxibj_I66_seed7|unseen_triple|16.494891|33.472697|
|taxibj_I66_seed7|unseen_combinations_repeat|17.934376|36.283587|
|taxibj_I66_seed7|unseen_geometry_repeat|19.637113|39.454159|
|taxibj_I67_seed7|in_distribution|8.526790|15.055947|
|taxibj_I67_seed7|unseen_combinations|19.325161|43.344558|
|taxibj_I67_seed7|unseen_geometry|20.643019|43.079533|
|taxibj_I67_seed7|unseen_triple|20.660745|41.469496|
|taxibj_I67_seed7|unseen_combinations_repeat|19.076153|42.442699|
|taxibj_I67_seed7|unseen_geometry_repeat|20.911347|43.884877|
|taxibj_I68_seed7|in_distribution|9.013286|16.145297|
|taxibj_I68_seed7|unseen_combinations|20.030127|39.836588|
|taxibj_I68_seed7|unseen_geometry|21.215333|43.594823|
|taxibj_I68_seed7|unseen_triple|25.637663|47.402900|
|taxibj_I68_seed7|unseen_combinations_repeat|20.021524|39.769719|
|taxibj_I68_seed7|unseen_geometry_repeat|21.512864|43.478689|
|taxibj_I69_seed7|in_distribution|8.432462|14.830133|
|taxibj_I69_seed7|unseen_combinations|26.960956|62.844714|
|taxibj_I69_seed7|unseen_geometry|24.199618|57.527665|
|taxibj_I69_seed7|unseen_triple|37.657826|74.288468|
|taxibj_I69_seed7|unseen_combinations_repeat|27.051783|62.747580|
|taxibj_I69_seed7|unseen_geometry_repeat|24.135823|56.696037|
|taxibj_I70_seed7|in_distribution|11.278834|20.035906|
|taxibj_I70_seed7|unseen_combinations|24.015073|46.337317|
|taxibj_I70_seed7|unseen_geometry|20.102841|40.520531|
|taxibj_I70_seed7|unseen_triple|18.054399|32.869163|
|taxibj_I70_seed7|unseen_combinations_repeat|24.028669|46.496780|
|taxibj_I70_seed7|unseen_geometry_repeat|20.309130|40.577462|
|taxibj_I71_seed7|in_distribution|8.616543|15.237946|
|taxibj_I71_seed7|unseen_combinations|21.585560|46.788573|
|taxibj_I71_seed7|unseen_geometry|19.803910|41.717895|
|taxibj_I71_seed7|unseen_triple|21.048852|44.666060|
|taxibj_I71_seed7|unseen_combinations_repeat|21.739533|46.943268|
|taxibj_I71_seed7|unseen_geometry_repeat|19.947568|41.885772|
|taxibj_I72_seed7|in_distribution|8.705907|15.512609|
|taxibj_I72_seed7|unseen_combinations|16.960428|33.705071|
|taxibj_I72_seed7|unseen_geometry|20.880806|43.020757|
|taxibj_I72_seed7|unseen_triple|16.058720|31.418577|
|taxibj_I72_seed7|unseen_combinations_repeat|16.695638|32.915572|
|taxibj_I72_seed7|unseen_geometry_repeat|20.842208|42.719090|
|taxibj_I73_seed7|in_distribution|8.514158|15.030622|
|taxibj_I73_seed7|unseen_combinations|14.271526|26.514846|
|taxibj_I73_seed7|unseen_geometry|18.029255|32.499178|
|taxibj_I73_seed7|unseen_triple|13.263602|24.474291|
|taxibj_I73_seed7|unseen_combinations_repeat|14.082930|25.919188|
|taxibj_I73_seed7|unseen_geometry_repeat|17.990586|32.767045|
|taxibj_I74_seed7|in_distribution|8.920183|15.826327|
|taxibj_I74_seed7|unseen_combinations|12.634235|22.254035|
|taxibj_I74_seed7|unseen_geometry|15.346775|28.227973|
|taxibj_I74_seed7|unseen_triple|11.500836|19.813536|
|taxibj_I74_seed7|unseen_combinations_repeat|12.675623|22.257604|
|taxibj_I74_seed7|unseen_geometry_repeat|15.409632|27.981626|
|taxibj_I75_seed7|in_distribution|10.984407|19.770276|
|taxibj_I75_seed7|unseen_combinations|24.195773|48.200714|
|taxibj_I75_seed7|unseen_geometry|23.286128|47.035915|
|taxibj_I75_seed7|unseen_triple|22.845008|45.505415|
|taxibj_I75_seed7|unseen_combinations_repeat|23.761327|47.429043|
|taxibj_I75_seed7|unseen_geometry_repeat|23.677102|47.582414|
### 六套验证开发协议

|模型|协议|MAE|RMSE|
|---|---|---:|---:|
|taxibj_I37_seed7|in_distribution|8.511569|15.997278|
|taxibj_I37_seed7|unseen_combinations|8.744007|15.622489|
|taxibj_I37_seed7|unseen_geometry|11.391335|19.576816|
|taxibj_I37_seed7|unseen_triple|8.363297|15.045544|
|taxibj_I37_seed7|unseen_combinations_repeat|8.661969|15.477529|
|taxibj_I37_seed7|unseen_geometry_repeat|11.495959|19.641001|
|taxibj_I56_seed7|in_distribution|10.928248|19.717644|
|taxibj_I56_seed7|unseen_combinations|17.129514|29.687354|
|taxibj_I56_seed7|unseen_geometry|18.801410|34.325755|
|taxibj_I56_seed7|unseen_triple|17.205685|29.452280|
|taxibj_I56_seed7|unseen_combinations_repeat|17.003331|29.357210|
|taxibj_I56_seed7|unseen_geometry_repeat|19.798363|36.189254|
|taxibj_I57_seed7|in_distribution|8.959466|16.700735|
|taxibj_I57_seed7|unseen_combinations|9.451022|16.864510|
|taxibj_I57_seed7|unseen_geometry|13.478054|25.230181|
|taxibj_I57_seed7|unseen_triple|9.041089|16.270800|
|taxibj_I57_seed7|unseen_combinations_repeat|9.316227|16.527085|
|taxibj_I57_seed7|unseen_geometry_repeat|13.591964|24.932501|
|taxibj_I64_seed7|in_distribution|8.542174|16.075222|
|taxibj_I64_seed7|unseen_combinations|20.584173|42.599370|
|taxibj_I64_seed7|unseen_geometry|23.312350|48.642300|
|taxibj_I64_seed7|unseen_triple|19.276517|41.351969|
|taxibj_I64_seed7|unseen_combinations_repeat|20.614722|42.372827|
|taxibj_I64_seed7|unseen_geometry_repeat|24.012853|49.470127|
|taxibj_I65_seed7|in_distribution|8.513877|16.028861|
|taxibj_I65_seed7|unseen_combinations|12.621597|25.400873|
|taxibj_I65_seed7|unseen_geometry|15.327238|28.818390|
|taxibj_I65_seed7|unseen_triple|11.335901|23.103129|
|taxibj_I65_seed7|unseen_combinations_repeat|12.168434|24.259245|
|taxibj_I65_seed7|unseen_geometry_repeat|15.510249|28.472191|
|taxibj_I66_seed7|in_distribution|8.720856|16.362066|
|taxibj_I66_seed7|unseen_combinations|18.854144|37.944803|
|taxibj_I66_seed7|unseen_geometry|20.845700|42.027934|
|taxibj_I66_seed7|unseen_triple|17.417407|35.124441|
|taxibj_I66_seed7|unseen_combinations_repeat|18.645324|37.270050|
|taxibj_I66_seed7|unseen_geometry_repeat|20.880927|41.935614|
|taxibj_I67_seed7|in_distribution|9.029045|16.499033|
|taxibj_I67_seed7|unseen_combinations|19.850559|43.283527|
|taxibj_I67_seed7|unseen_geometry|22.001470|46.511024|
|taxibj_I67_seed7|unseen_triple|22.284690|44.788742|
|taxibj_I67_seed7|unseen_combinations_repeat|19.968028|43.356168|
|taxibj_I67_seed7|unseen_geometry_repeat|21.745890|45.509921|
|taxibj_I68_seed7|in_distribution|9.586366|17.796266|
|taxibj_I68_seed7|unseen_combinations|21.156508|42.057280|
|taxibj_I68_seed7|unseen_geometry|22.624801|46.201896|
|taxibj_I68_seed7|unseen_triple|27.254490|50.436710|
|taxibj_I68_seed7|unseen_combinations_repeat|20.834118|41.245789|
|taxibj_I68_seed7|unseen_geometry_repeat|24.299436|48.904265|
|taxibj_I69_seed7|in_distribution|9.062476|16.620731|
|taxibj_I69_seed7|unseen_combinations|28.730169|66.273680|
|taxibj_I69_seed7|unseen_geometry|25.372225|60.504471|
|taxibj_I69_seed7|unseen_triple|40.263679|78.944896|
|taxibj_I69_seed7|unseen_combinations_repeat|28.286639|64.954721|
|taxibj_I69_seed7|unseen_geometry_repeat|28.320556|66.589581|
|taxibj_I70_seed7|in_distribution|12.093179|21.809483|
|taxibj_I70_seed7|unseen_combinations|26.332876|51.851770|
|taxibj_I70_seed7|unseen_geometry|20.932607|41.729073|
|taxibj_I70_seed7|unseen_triple|19.578612|36.000841|
|taxibj_I70_seed7|unseen_combinations_repeat|24.615506|47.076173|
|taxibj_I70_seed7|unseen_geometry_repeat|22.059212|43.940763|
|taxibj_I71_seed7|in_distribution|9.313337|17.181727|
|taxibj_I71_seed7|unseen_combinations|23.530335|50.957223|
|taxibj_I71_seed7|unseen_geometry|21.389778|45.549641|
|taxibj_I71_seed7|unseen_triple|23.297131|49.299613|
|taxibj_I71_seed7|unseen_combinations_repeat|23.028469|49.416499|
|taxibj_I71_seed7|unseen_geometry_repeat|22.265274|46.385733|
|taxibj_I72_seed7|in_distribution|9.377216|17.389362|
|taxibj_I72_seed7|unseen_combinations|17.964575|35.633808|
|taxibj_I72_seed7|unseen_geometry|21.431765|44.053298|
|taxibj_I72_seed7|unseen_triple|16.893854|32.726635|
|taxibj_I72_seed7|unseen_combinations_repeat|17.610232|34.436351|
|taxibj_I72_seed7|unseen_geometry_repeat|22.431694|46.027533|
|taxibj_I73_seed7|in_distribution|9.066105|16.441448|
|taxibj_I73_seed7|unseen_combinations|14.984809|28.068610|
|taxibj_I73_seed7|unseen_geometry|18.721767|33.861888|
|taxibj_I73_seed7|unseen_triple|13.740440|25.126663|
|taxibj_I73_seed7|unseen_combinations_repeat|14.598007|27.295914|
|taxibj_I73_seed7|unseen_geometry_repeat|18.673994|33.371348|
|taxibj_I74_seed7|in_distribution|9.462256|17.333391|
|taxibj_I74_seed7|unseen_combinations|13.240880|23.807733|
|taxibj_I74_seed7|unseen_geometry|16.025357|29.349123|
|taxibj_I74_seed7|unseen_triple|12.077394|20.949707|
|taxibj_I74_seed7|unseen_combinations_repeat|13.156217|23.175513|
|taxibj_I74_seed7|unseen_geometry_repeat|16.042382|29.215509|
|taxibj_I75_seed7|in_distribution|11.808977|21.738472|
|taxibj_I75_seed7|unseen_combinations|25.808645|51.392249|
|taxibj_I75_seed7|unseen_geometry|24.968027|51.591671|
|taxibj_I75_seed7|unseen_triple|25.145706|49.993300|
|taxibj_I75_seed7|unseen_combinations_repeat|25.127616|49.438807|
|taxibj_I75_seed7|unseen_geometry_repeat|26.307915|52.935716|
### 新mask确认协议

|模型|协议|MAE|RMSE|
|---|---|---:|---:|
|taxibj_I37_seed7|in_distribution|7.980677|14.461614|
|taxibj_I37_seed7|unseen_combinations|8.200258|14.510607|
|taxibj_I37_seed7|unseen_geometry|10.712666|18.174050|
|taxibj_I37_seed7|unseen_triple|7.972974|14.126859|
|taxibj_I37_seed7|unseen_combinations_repeat|8.167178|14.344431|
|taxibj_I37_seed7|unseen_geometry_repeat|10.707337|18.188320|
|taxibj_I56_seed7|in_distribution|10.239052|18.006749|
|taxibj_I56_seed7|unseen_combinations|16.183929|27.887569|
|taxibj_I56_seed7|unseen_geometry|18.074193|32.973945|
|taxibj_I56_seed7|unseen_triple|16.384736|27.983460|
|taxibj_I56_seed7|unseen_combinations_repeat|16.105478|27.804777|
|taxibj_I56_seed7|unseen_geometry_repeat|17.794024|32.363320|
|taxibj_I57_seed7|in_distribution|8.318989|14.909576|
|taxibj_I57_seed7|unseen_combinations|8.912836|15.795948|
|taxibj_I57_seed7|unseen_geometry|12.793886|23.779664|
|taxibj_I57_seed7|unseen_triple|8.640819|15.293246|
|taxibj_I57_seed7|unseen_combinations_repeat|8.885111|15.582963|
|taxibj_I57_seed7|unseen_geometry_repeat|12.734648|23.763555|
|taxibj_I64_seed7|in_distribution|8.010975|14.494138|
|taxibj_I64_seed7|unseen_combinations|19.875693|41.534805|
|taxibj_I64_seed7|unseen_geometry|21.832511|44.625550|
|taxibj_I64_seed7|unseen_triple|18.457261|40.067889|
|taxibj_I64_seed7|unseen_combinations_repeat|19.339262|39.418551|
|taxibj_I64_seed7|unseen_geometry_repeat|21.660530|45.022577|
|taxibj_I65_seed7|in_distribution|7.887882|14.164714|
|taxibj_I65_seed7|unseen_combinations|11.804147|23.548561|
|taxibj_I65_seed7|unseen_geometry|14.804941|27.682700|
|taxibj_I65_seed7|unseen_triple|10.872660|22.077835|
|taxibj_I65_seed7|unseen_combinations_repeat|11.498068|22.695915|
|taxibj_I65_seed7|unseen_geometry_repeat|14.654126|27.776397|
|taxibj_I66_seed7|in_distribution|8.159952|14.644761|
|taxibj_I66_seed7|unseen_combinations|18.042457|36.381212|
|taxibj_I66_seed7|unseen_geometry|19.623063|38.910450|
|taxibj_I66_seed7|unseen_triple|16.744238|33.935435|
|taxibj_I66_seed7|unseen_combinations_repeat|17.512112|34.873913|
|taxibj_I66_seed7|unseen_geometry_repeat|19.429634|39.334631|
|taxibj_I67_seed7|in_distribution|8.547077|15.236914|
|taxibj_I67_seed7|unseen_combinations|19.447840|44.604390|
|taxibj_I67_seed7|unseen_geometry|20.618570|42.591769|
|taxibj_I67_seed7|unseen_triple|21.339515|43.399336|
|taxibj_I67_seed7|unseen_combinations_repeat|18.637964|41.377998|
|taxibj_I67_seed7|unseen_geometry_repeat|20.348298|43.297181|
|taxibj_I68_seed7|in_distribution|9.001621|16.135459|
|taxibj_I68_seed7|unseen_combinations|19.660643|39.078112|
|taxibj_I68_seed7|unseen_geometry|21.186053|43.466264|
|taxibj_I68_seed7|unseen_triple|25.607485|47.526150|
|taxibj_I68_seed7|unseen_combinations_repeat|19.430066|38.431457|
|taxibj_I68_seed7|unseen_geometry_repeat|21.084405|43.325439|
|taxibj_I69_seed7|in_distribution|8.461164|15.026075|
|taxibj_I69_seed7|unseen_combinations|26.655351|62.005171|
|taxibj_I69_seed7|unseen_geometry|24.076436|57.381604|
|taxibj_I69_seed7|unseen_triple|38.127986|75.831180|
|taxibj_I69_seed7|unseen_combinations_repeat|26.182806|60.690785|
|taxibj_I69_seed7|unseen_geometry_repeat|23.819522|56.831961|
|taxibj_I70_seed7|in_distribution|11.264477|19.988615|
|taxibj_I70_seed7|unseen_combinations|24.257473|47.292826|
|taxibj_I70_seed7|unseen_geometry|20.006811|39.816888|
|taxibj_I70_seed7|unseen_triple|18.300634|33.542528|
|taxibj_I70_seed7|unseen_combinations_repeat|23.428050|45.342002|
|taxibj_I70_seed7|unseen_geometry_repeat|20.352049|41.884888|
|taxibj_I71_seed7|in_distribution|8.638634|15.321476|
|taxibj_I71_seed7|unseen_combinations|21.568806|46.817884|
|taxibj_I71_seed7|unseen_geometry|20.114423|43.050612|
|taxibj_I71_seed7|unseen_triple|21.678185|46.400691|
|taxibj_I71_seed7|unseen_combinations_repeat|20.846419|45.306800|
|taxibj_I71_seed7|unseen_geometry_repeat|19.981270|42.710507|
|taxibj_I72_seed7|in_distribution|8.681397|15.523172|
|taxibj_I72_seed7|unseen_combinations|16.842380|33.412966|
|taxibj_I72_seed7|unseen_geometry|20.519715|42.087598|
|taxibj_I72_seed7|unseen_triple|16.227931|31.776907|
|taxibj_I72_seed7|unseen_combinations_repeat|16.582079|32.711623|
|taxibj_I72_seed7|unseen_geometry_repeat|20.375472|42.411870|
|taxibj_I73_seed7|in_distribution|8.476249|14.931491|
|taxibj_I73_seed7|unseen_combinations|14.110973|26.278390|
|taxibj_I73_seed7|unseen_geometry|17.975738|32.380563|
|taxibj_I73_seed7|unseen_triple|13.353454|24.688879|
|taxibj_I73_seed7|unseen_combinations_repeat|13.913349|25.271762|
|taxibj_I73_seed7|unseen_geometry_repeat|17.582226|31.713265|
|taxibj_I74_seed7|in_distribution|8.858644|15.765266|
|taxibj_I74_seed7|unseen_combinations|12.501955|21.948605|
|taxibj_I74_seed7|unseen_geometry|15.308534|28.076415|
|taxibj_I74_seed7|unseen_triple|11.609326|20.140611|
|taxibj_I74_seed7|unseen_combinations_repeat|12.416977|21.842740|
|taxibj_I74_seed7|unseen_geometry_repeat|15.111167|28.009057|
|taxibj_I75_seed7|in_distribution|11.017869|19.991268|
|taxibj_I75_seed7|unseen_combinations|23.732026|47.328393|
|taxibj_I75_seed7|unseen_geometry|23.146351|47.218385|
|taxibj_I75_seed7|unseen_triple|23.495511|47.031545|
|taxibj_I75_seed7|unseen_combinations_repeat|23.365750|46.337998|
|taxibj_I75_seed7|unseen_geometry_repeat|23.318109|47.804702|

### 训练完成情况、成本与曲线

|组|完成/best epoch|best验证MAE|参数/可训练参数|训练小时|峰值GiB|best/last|GPU日志|
|---|---|---:|---|---:|---:|---|---|
|taxibj_I64_seed7|100/90|8.262863|347930/333019|1.301|12.889|有|有|
|taxibj_I65_seed7|100/100|8.171537|347930/333019|1.779|11.229|有|有|
|taxibj_I66_seed7|100/90|8.415990|347930/333019|1.566|12.838|有|有|
|taxibj_I67_seed7|100/100|8.760315|347930/333019|1.815|12.954|有|有|
|taxibj_I68_seed7|100/95|9.311928|347930/333019|1.291|11.561|有|有|
|taxibj_I69_seed7|100/90|8.782850|347930/333019|1.235|12.954|有|有|
|taxibj_I70_seed7|100/95|11.667164|347930/333019|1.393|11.337|有|有|
|taxibj_I71_seed7|100/100|8.954201|347930/333019|1.187|11.312|有|有|
|taxibj_I72_seed7|100/100|9.048156|347930/333019|1.443|11.332|有|有|
|taxibj_I73_seed7|100/95|8.783523|347930/333019|0.910|11.266|有|有|
|taxibj_I74_seed7|100/100|9.105913|347930/333019|1.402|12.818|有|有|
|taxibj_I75_seed7|100/100|11.427534|347930/333019|1.616|12.908|有|有|

置零输入通道仍保留并计算对应网络参数，参数相同不代表有效容量或梯度参与相同；不据此宣称FLOPs减少。峰值显存及耗时受实际路由影响。

taxibj_I64_seed7：loss 95.2675→11.4952；末轮ID验证MAE 8.4306，best在90轮。

![taxibj_I64_seed7](figures/taxibj_I64_seed7.png)

taxibj_I65_seed7：loss 95.9401→10.7541；末轮ID验证MAE 8.1715，best在100轮。

![taxibj_I65_seed7](figures/taxibj_I65_seed7.png)

taxibj_I66_seed7：loss 95.4353→11.1343；末轮ID验证MAE 8.4749，best在90轮。

![taxibj_I66_seed7](figures/taxibj_I66_seed7.png)

taxibj_I67_seed7：loss 96.4192→11.7154；末轮ID验证MAE 8.7603，best在100轮。

![taxibj_I67_seed7](figures/taxibj_I67_seed7.png)

taxibj_I68_seed7：loss 97.5247→12.1888；末轮ID验证MAE 9.4661，best在95轮。

![taxibj_I68_seed7](figures/taxibj_I68_seed7.png)

taxibj_I69_seed7：loss 95.3508→12.2290；末轮ID验证MAE 8.8197，best在90轮。

![taxibj_I69_seed7](figures/taxibj_I69_seed7.png)

taxibj_I70_seed7：loss 96.0221→15.4529；末轮ID验证MAE 11.9020，best在95轮。

![taxibj_I70_seed7](figures/taxibj_I70_seed7.png)

taxibj_I71_seed7：loss 96.1866→12.3067；末轮ID验证MAE 8.9542，best在100轮。

![taxibj_I71_seed7](figures/taxibj_I71_seed7.png)

taxibj_I72_seed7：loss 96.0164→11.9058；末轮ID验证MAE 9.0482，best在100轮。

![taxibj_I72_seed7](figures/taxibj_I72_seed7.png)

taxibj_I73_seed7：loss 97.4586→11.6325；末轮ID验证MAE 9.0315，best在95轮。

![taxibj_I73_seed7](figures/taxibj_I73_seed7.png)

taxibj_I74_seed7：loss 96.6188→12.2832；末轮ID验证MAE 9.1059，best在100轮。

![taxibj_I74_seed7](figures/taxibj_I74_seed7.png)

taxibj_I75_seed7：loss 96.9733→15.6971；末轮ID验证MAE 11.4275，best在100轮。

![taxibj_I75_seed7](figures/taxibj_I75_seed7.png)

### 输入及投影前后诊断

|模型|协议|投影前F MAE|投影后F MAE|约束区域MAE/计数|无约束区域MAE/计数|末轮fallback幅度|
|---|---|---:|---:|---|---|---|
|taxibj_I37_seed7|in_distribution|11.09543|8.05104|6.65282/5658230|14.17645/1291580|—|
|taxibj_I37_seed7|unseen_geometry|13.49171|10.82116|8.60402/5463710|18.97256/1486100|—|
|taxibj_I37_seed7|unseen_triple|11.07618|7.89207|6.71979/5809322|13.86336/1140488|—|
|taxibj_I56_seed7|in_distribution|13.57701|10.29672|8.82857/5658230|16.72847/1291580|0.0|
|taxibj_I56_seed7|unseen_geometry|21.98016|18.07950|14.53800/5463710|31.09999/1486100|0.0|
|taxibj_I56_seed7|unseen_triple|22.24052|16.29294|14.25249/5809322|26.68642/1140488|0.0|
|taxibj_I57_seed7|in_distribution|11.24526|8.39925|6.96719/5658230|14.67288/1291580|0.0|
|taxibj_I57_seed7|unseen_geometry|15.26688|12.99449|9.57793/5463710|25.55563/1486100|0.0|
|taxibj_I57_seed7|unseen_triple|11.83087|8.53208|7.27389/5809322|14.94095/1140488|0.0|
|taxibj_I64_seed7|in_distribution|10.97306|8.08259|6.70708/5658230|14.10851/1291580|54.21681744119395|
|taxibj_I64_seed7|unseen_geometry|28.12739|22.04208|16.13786/5463710|43.74920/1486100|31.524731179942254|
|taxibj_I64_seed7|unseen_triple|28.24688|18.25792|15.00174/5809322|34.84397/1140488|69.68124638433042|
|taxibj_I65_seed7|in_distribution|10.85357|7.88539|6.59620/5658230|13.53315/1291580|20.63227889848792|
|taxibj_I65_seed7|unseen_geometry|19.11085|14.84953|10.88105/5463710|29.43981/1486100|18.8290898281595|
|taxibj_I65_seed7|unseen_triple|15.95156|10.70340|8.40615/5809322|22.40495/1140488|21.609263171320375|
|taxibj_I66_seed7|in_distribution|11.32048|8.15370|6.77779/5658230|14.18137/1291580|36.70016695105511|
|taxibj_I66_seed7|unseen_geometry|25.92673|19.61221|14.24823/5463710|39.33309/1486100|20.826192607050356|
|taxibj_I66_seed7|unseen_triple|25.07275|16.49489|13.54328/5809322|31.52958/1140488|48.08179473876953|
|taxibj_I67_seed7|in_distribution|12.26237|8.52679|7.17430/5658230|14.45188/1291580|0.0|
|taxibj_I67_seed7|unseen_geometry|27.78617|20.64302|15.21413/5463710|40.60257/1486100|0.0|
|taxibj_I67_seed7|unseen_triple|33.61909|20.66075|16.86157/5809322|40.01265/1140488|0.0|
|taxibj_I68_seed7|in_distribution|11.85901|9.01329|7.53879/5658230|15.47286/1291580|0.0|
|taxibj_I68_seed7|unseen_geometry|27.09717|21.21533|17.26244/5463710|35.74831/1486100|0.0|
|taxibj_I68_seed7|unseen_triple|42.33030|25.63766|21.65737/5809322|45.91216/1140488|0.0|
|taxibj_I69_seed7|in_distribution|11.66560|8.43246|7.17684/5658230|13.93317/1291580|0.0|
|taxibj_I69_seed7|unseen_geometry|30.52639|24.19962|20.43632/5463710|38.03555/1486100|0.0|
|taxibj_I69_seed7|unseen_triple|61.99798|37.65783|32.24946/5809322|65.20652/1140488|0.0|
|taxibj_I70_seed7|in_distribution|15.05732|11.27883|9.73168/5658230|18.05668/1291580|83.46958492113197|
|taxibj_I70_seed7|unseen_geometry|26.74475|20.10284|15.25245/5463710|37.93553/1486100|35.63342144178308|
|taxibj_I70_seed7|unseen_triple|26.90983|18.05440|14.93365/5809322|33.95061/1140488|113.67828933052395|
|taxibj_I71_seed7|in_distribution|11.90319|8.61654|7.29377/5658230|14.41141/1291580|0.0|
|taxibj_I71_seed7|unseen_geometry|25.60218|19.80391|15.83790/5463710|34.38511/1486100|0.0|
|taxibj_I71_seed7|unseen_triple|31.86039|21.04885|18.21382/5809322|35.48969/1140488|0.0|
|taxibj_I72_seed7|in_distribution|11.38487|8.70591|7.34912/5658230|14.64978/1291580|45.307638500047766|
|taxibj_I72_seed7|unseen_geometry|26.94714|20.88081|15.15146/5463710|41.94500/1486100|42.41738211590311|
|taxibj_I72_seed7|unseen_triple|23.58375|16.05872|13.03883/5809322|31.44118/1140488|47.841819348542586|
|taxibj_I73_seed7|in_distribution|11.59927|8.51416|7.17172/5658230|14.39519/1291580|70.38203197976817|
|taxibj_I73_seed7|unseen_geometry|22.10267|18.02926|13.74810/5463710|33.76911/1486100|67.69847305961277|
|taxibj_I73_seed7|unseen_triple|19.16569|13.26360|10.78400/5809322|25.89397/1140488|74.96245790564495|
|taxibj_I74_seed7|in_distribution|12.12575|8.92018|7.55827/5658230|14.88655/1291580|0.0|
|taxibj_I74_seed7|unseen_geometry|19.02062|15.34677|12.20271/5463710|26.90607/1486100|0.0|
|taxibj_I74_seed7|unseen_triple|16.40144|11.50084|9.93901/5809322|19.45635/1140488|0.0|
|taxibj_I75_seed7|in_distribution|15.24381|10.98441|9.40612/5658230|17.89866/1291580|3.158447540324667|
|taxibj_I75_seed7|unseen_geometry|30.65111|23.28613|18.48979/5463710|40.92006/1486100|1.6870474763538525|
|taxibj_I75_seed7|unseen_triple|34.08218|22.84501|18.82796/5809322|43.30670/1140488|3.432521747506183|

### 补位来源与重复元数据分量

|模型|协议|轮次|可见F补位贡献幅度|P0补位贡献幅度|support RMS|position RMS|
|---|---|---:|---:|---:|---:|---:|
|taxibj_I64_seed7|in_distribution|1|5.983702|13.148626|0.000000|0.302246|
|taxibj_I64_seed7|in_distribution|2|53.431372|26.373983|0.000000|0.302093|
|taxibj_I64_seed7|in_distribution|3|3.363853|91.014469|0.000000|0.302316|
|taxibj_I64_seed7|in_distribution|4|0.000000|54.216817|0.000000|0.302418|
|taxibj_I64_seed7|unseen_geometry|1|3.367402|13.505609|0.000000|0.302335|
|taxibj_I64_seed7|unseen_geometry|2|54.578312|13.999339|0.000000|0.302093|
|taxibj_I64_seed7|unseen_geometry|3|9.369855|58.488681|0.000000|0.302113|
|taxibj_I64_seed7|unseen_geometry|4|0.000000|31.524731|0.000000|0.302418|
|taxibj_I64_seed7|unseen_triple|1|49.818987|29.330436|0.000000|0.301018|
|taxibj_I64_seed7|unseen_triple|2|55.028527|31.286259|0.000000|0.302093|
|taxibj_I64_seed7|unseen_triple|3|0.165034|72.900912|0.000000|0.302407|
|taxibj_I64_seed7|unseen_triple|4|0.000000|69.681246|0.000000|0.302418|
|taxibj_I65_seed7|in_distribution|1|0.676388|22.074038|0.539096|0.000000|
|taxibj_I65_seed7|in_distribution|2|54.054824|8.370971|0.503413|0.000000|
|taxibj_I65_seed7|in_distribution|3|0.000000|20.632279|0.511392|0.000000|
|taxibj_I65_seed7|in_distribution|4|0.000000|20.632279|0.511392|0.000000|
|taxibj_I65_seed7|unseen_geometry|1|0.323291|18.936720|0.522981|0.000000|
|taxibj_I65_seed7|unseen_geometry|2|54.032689|7.762916|0.509747|0.000000|
|taxibj_I65_seed7|unseen_geometry|3|0.000000|18.829090|0.519227|0.000000|
|taxibj_I65_seed7|unseen_geometry|4|0.000000|18.829090|0.519227|0.000000|
|taxibj_I65_seed7|unseen_triple|1|0.000000|21.609263|0.551490|0.000000|
|taxibj_I65_seed7|unseen_triple|2|56.143281|8.549441|0.537239|0.000000|
|taxibj_I65_seed7|unseen_triple|3|0.000000|21.609263|0.551490|0.000000|
|taxibj_I65_seed7|unseen_triple|4|0.000000|21.609263|0.551490|0.000000|
|taxibj_I66_seed7|in_distribution|1|54.054824|18.486820|0.000000|0.000000|
|taxibj_I66_seed7|in_distribution|2|1.449697|41.206549|0.000000|0.000000|
|taxibj_I66_seed7|in_distribution|3|0.017722|36.795693|0.000000|0.000000|
|taxibj_I66_seed7|in_distribution|4|0.000000|36.700167|0.000000|0.000000|
|taxibj_I66_seed7|unseen_geometry|1|54.032689|10.609874|0.000000|0.000000|
|taxibj_I66_seed7|unseen_geometry|2|2.165481|23.942585|0.000000|0.000000|
|taxibj_I66_seed7|unseen_geometry|3|0.025102|20.912762|0.000000|0.000000|
|taxibj_I66_seed7|unseen_geometry|4|0.000000|20.826193|0.000000|0.000000|
|taxibj_I66_seed7|unseen_triple|1|56.143281|21.342613|0.000000|0.000000|
|taxibj_I66_seed7|unseen_triple|2|0.012819|48.151728|0.000000|0.000000|
|taxibj_I66_seed7|unseen_triple|3|0.000000|48.081795|0.000000|0.000000|
|taxibj_I66_seed7|unseen_triple|4|0.000000|48.081795|0.000000|0.000000|
|taxibj_I67_seed7|in_distribution|1|0.000000|0.000000|0.503413|0.000000|
|taxibj_I67_seed7|in_distribution|2|0.000000|0.000000|0.511392|0.000000|
|taxibj_I67_seed7|in_distribution|3|0.000000|0.000000|0.511392|0.000000|
|taxibj_I67_seed7|in_distribution|4|0.000000|0.000000|0.511392|0.000000|
|taxibj_I67_seed7|unseen_geometry|1|0.000000|0.000000|0.509747|0.000000|
|taxibj_I67_seed7|unseen_geometry|2|0.000000|0.000000|0.519227|0.000000|
|taxibj_I67_seed7|unseen_geometry|3|0.000000|0.000000|0.519227|0.000000|
|taxibj_I67_seed7|unseen_geometry|4|0.000000|0.000000|0.519227|0.000000|
|taxibj_I67_seed7|unseen_triple|1|0.000000|0.000000|0.537239|0.000000|
|taxibj_I67_seed7|unseen_triple|2|0.000000|0.000000|0.551490|0.000000|
|taxibj_I67_seed7|unseen_triple|3|0.000000|0.000000|0.551490|0.000000|
|taxibj_I67_seed7|unseen_triple|4|0.000000|0.000000|0.551490|0.000000|
|taxibj_I68_seed7|in_distribution|1|0.000000|0.000000|0.000000|0.302113|
|taxibj_I68_seed7|in_distribution|2|0.000000|0.000000|0.000000|0.301750|
|taxibj_I68_seed7|in_distribution|3|0.000000|0.000000|0.000000|0.302113|
|taxibj_I68_seed7|in_distribution|4|0.000000|0.000000|0.000000|0.302113|
|taxibj_I68_seed7|unseen_geometry|1|0.000000|0.000000|0.000000|0.302113|
|taxibj_I68_seed7|unseen_geometry|2|0.000000|0.000000|0.000000|0.301752|
|taxibj_I68_seed7|unseen_geometry|3|0.000000|0.000000|0.000000|0.302113|
|taxibj_I68_seed7|unseen_geometry|4|0.000000|0.000000|0.000000|0.302113|
|taxibj_I68_seed7|unseen_triple|1|0.000000|0.000000|0.000000|0.302113|
|taxibj_I68_seed7|unseen_triple|2|0.000000|0.000000|0.000000|0.301758|
|taxibj_I68_seed7|unseen_triple|3|0.000000|0.000000|0.000000|0.302113|
|taxibj_I68_seed7|unseen_triple|4|0.000000|0.000000|0.000000|0.302113|
|taxibj_I69_seed7|in_distribution|1|0.000000|0.000000|0.000000|0.000000|
|taxibj_I69_seed7|in_distribution|2|0.000000|0.000000|0.000000|0.000000|
|taxibj_I69_seed7|in_distribution|3|0.000000|0.000000|0.000000|0.000000|
|taxibj_I69_seed7|in_distribution|4|0.000000|0.000000|0.000000|0.000000|
|taxibj_I69_seed7|unseen_geometry|1|0.000000|0.000000|0.000000|0.000000|
|taxibj_I69_seed7|unseen_geometry|2|0.000000|0.000000|0.000000|0.000000|
|taxibj_I69_seed7|unseen_geometry|3|0.000000|0.000000|0.000000|0.000000|
|taxibj_I69_seed7|unseen_geometry|4|0.000000|0.000000|0.000000|0.000000|
|taxibj_I69_seed7|unseen_triple|1|0.000000|0.000000|0.000000|0.000000|
|taxibj_I69_seed7|unseen_triple|2|0.000000|0.000000|0.000000|0.000000|
|taxibj_I69_seed7|unseen_triple|3|0.000000|0.000000|0.000000|0.000000|
|taxibj_I69_seed7|unseen_triple|4|0.000000|0.000000|0.000000|0.000000|
|taxibj_I70_seed7|in_distribution|1|0.001320|83.619856|0.000000|0.000000|
|taxibj_I70_seed7|in_distribution|2|53.431372|38.270308|0.000000|0.000000|
|taxibj_I70_seed7|in_distribution|3|54.054824|37.366122|0.000000|0.000000|
|taxibj_I70_seed7|in_distribution|4|0.000000|83.469585|0.000000|0.000000|
|taxibj_I70_seed7|unseen_geometry|1|0.000000|35.633421|0.000000|0.000000|
|taxibj_I70_seed7|unseen_geometry|2|54.578312|16.533626|0.000000|0.000000|
|taxibj_I70_seed7|unseen_geometry|3|54.032689|16.918976|0.000000|0.000000|
|taxibj_I70_seed7|unseen_geometry|4|0.000000|35.633421|0.000000|0.000000|
|taxibj_I70_seed7|unseen_triple|1|0.000000|113.678289|0.000000|0.000000|
|taxibj_I70_seed7|unseen_triple|2|55.028527|49.179067|0.000000|0.000000|
|taxibj_I70_seed7|unseen_triple|3|56.143281|47.149067|0.000000|0.000000|
|taxibj_I70_seed7|unseen_triple|4|0.000000|113.678289|0.000000|0.000000|
|taxibj_I71_seed7|in_distribution|1|0.000000|0.000000|0.000000|0.000000|
|taxibj_I71_seed7|in_distribution|2|53.431372|0.000000|0.000000|0.000000|
|taxibj_I71_seed7|in_distribution|3|54.054824|0.000000|0.000000|0.000000|
|taxibj_I71_seed7|in_distribution|4|0.000000|0.000000|0.000000|0.000000|
|taxibj_I71_seed7|unseen_geometry|1|0.000000|0.000000|0.000000|0.000000|
|taxibj_I71_seed7|unseen_geometry|2|54.578312|0.000000|0.000000|0.000000|
|taxibj_I71_seed7|unseen_geometry|3|54.032689|0.000000|0.000000|0.000000|
|taxibj_I71_seed7|unseen_geometry|4|0.000000|0.000000|0.000000|0.000000|
|taxibj_I71_seed7|unseen_triple|1|0.000000|0.000000|0.000000|0.000000|
|taxibj_I71_seed7|unseen_triple|2|55.028527|0.000000|0.000000|0.000000|
|taxibj_I71_seed7|unseen_triple|3|56.143281|0.000000|0.000000|0.000000|
|taxibj_I71_seed7|unseen_triple|4|0.000000|0.000000|0.000000|0.000000|
|taxibj_I72_seed7|in_distribution|1|0.000000|45.307639|0.000000|0.000000|
|taxibj_I72_seed7|in_distribution|2|53.431372|17.376713|0.000000|0.000000|
|taxibj_I72_seed7|in_distribution|3|0.000000|45.307639|0.000000|0.000000|
|taxibj_I72_seed7|in_distribution|4|0.000000|45.307639|0.000000|0.000000|
|taxibj_I72_seed7|unseen_geometry|1|0.000000|42.417382|0.000000|0.000000|
|taxibj_I72_seed7|unseen_geometry|2|54.578312|15.929459|0.000000|0.000000|
|taxibj_I72_seed7|unseen_geometry|3|0.000000|42.417382|0.000000|0.000000|
|taxibj_I72_seed7|unseen_geometry|4|0.000000|42.417382|0.000000|0.000000|
|taxibj_I72_seed7|unseen_triple|1|0.000000|47.841819|0.000000|0.000000|
|taxibj_I72_seed7|unseen_triple|2|55.028527|17.651829|0.000000|0.000000|
|taxibj_I72_seed7|unseen_triple|3|0.000000|47.841819|0.000000|0.000000|
|taxibj_I72_seed7|unseen_triple|4|0.000000|47.841819|0.000000|0.000000|
|taxibj_I73_seed7|in_distribution|1|54.054824|31.510276|0.000000|0.000000|
|taxibj_I73_seed7|in_distribution|2|53.431372|31.890347|0.000000|0.000000|
|taxibj_I73_seed7|in_distribution|3|54.054824|31.510276|0.000000|0.000000|
|taxibj_I73_seed7|in_distribution|4|0.000000|70.382032|0.000000|0.000000|
|taxibj_I73_seed7|unseen_geometry|1|54.032689|33.095084|0.000000|0.000000|
|taxibj_I73_seed7|unseen_geometry|2|54.578312|31.986845|0.000000|0.000000|
|taxibj_I73_seed7|unseen_geometry|3|54.032689|33.095084|0.000000|0.000000|
|taxibj_I73_seed7|unseen_geometry|4|0.000000|67.698473|0.000000|0.000000|
|taxibj_I73_seed7|unseen_triple|1|56.143281|31.753234|0.000000|0.000000|
|taxibj_I73_seed7|unseen_triple|2|55.028527|32.795278|0.000000|0.000000|
|taxibj_I73_seed7|unseen_triple|3|56.143281|31.753234|0.000000|0.000000|
|taxibj_I73_seed7|unseen_triple|4|0.000000|74.962458|0.000000|0.000000|
|taxibj_I74_seed7|in_distribution|1|53.431372|0.000000|0.127045|0.095357|
|taxibj_I74_seed7|in_distribution|2|46.626595|0.000000|0.130461|0.094977|
|taxibj_I74_seed7|in_distribution|3|0.000000|0.000000|0.127848|0.095478|
|taxibj_I74_seed7|in_distribution|4|0.000000|0.000000|0.127848|0.095478|
|taxibj_I74_seed7|unseen_geometry|1|54.578312|0.000000|0.128833|0.095357|
|taxibj_I74_seed7|unseen_geometry|2|49.123096|0.000000|0.128973|0.094950|
|taxibj_I74_seed7|unseen_geometry|3|0.000000|0.000000|0.129807|0.095478|
|taxibj_I74_seed7|unseen_geometry|4|0.000000|0.000000|0.129807|0.095478|
|taxibj_I74_seed7|unseen_triple|1|55.028527|0.000000|0.136884|0.095357|
|taxibj_I74_seed7|unseen_triple|2|52.470526|0.000000|0.134895|0.094932|
|taxibj_I74_seed7|unseen_triple|3|0.000000|0.000000|0.137873|0.095478|
|taxibj_I74_seed7|unseen_triple|4|0.000000|0.000000|0.137873|0.095478|
|taxibj_I75_seed7|in_distribution|1|53.431372|1.551522|0.127045|0.081168|
|taxibj_I75_seed7|in_distribution|2|0.000000|3.158448|0.127848|0.081260|
|taxibj_I75_seed7|in_distribution|3|12.913378|1.232248|0.111961|0.081238|
|taxibj_I75_seed7|in_distribution|4|0.000000|3.158448|0.127848|0.081260|
|taxibj_I75_seed7|unseen_geometry|1|54.578312|0.741749|0.128833|0.081168|
|taxibj_I75_seed7|unseen_geometry|2|0.000000|1.687047|0.129807|0.081260|
|taxibj_I75_seed7|unseen_geometry|3|3.806589|1.289556|0.125787|0.081252|
|taxibj_I75_seed7|unseen_geometry|4|0.000000|1.687047|0.129807|0.081260|
|taxibj_I75_seed7|unseen_triple|1|55.028527|1.422970|0.136884|0.081168|
|taxibj_I75_seed7|unseen_triple|2|0.000000|3.432522|0.137873|0.081260|
|taxibj_I75_seed7|unseen_triple|3|35.754702|2.259508|0.136878|0.081194|
|taxibj_I75_seed7|unseen_triple|4|0.000000|3.432522|0.137873|0.081260|

每轮fallback、有效观测计数、尺度路径、专家使用、历史/差分强度和投影残差完整保留于[诊断JSON（Git归档为压缩指标）](evaluations/)。fallback统计包含可见F的聚合贡献，不能统一理解为纯预测幅度。投影前指标读取raw预测，不使用已被覆盖的末轮输出。

### 冻结开发选择记录

```json
{
  "selected": "I65",
  "rows": [
    {
      "method": "I65",
      "id_mae": 8.51387687916808,
      "id_ratio": 1.0002711455680084,
      "eligible": true,
      "ood_mae_ratio": 1.3756741413983278,
      "ood_rmse_ratio": 1.531054726235664,
      "parameters": 347930
    },
    {
      "method": "I74",
      "id_mae": 9.462255864313208,
      "id_ratio": 1.1116934913884986,
      "eligible": false,
      "ood_mae_ratio": 1.453921252931116,
      "ood_rmse_ratio": 1.4654836204790342,
      "parameters": 347930
    },
    {
      "method": "I73",
      "id_mae": 9.06610503560529,
      "id_ratio": 1.0651508588283594,
      "eligible": false,
      "ood_mae_ratio": 1.658810849529598,
      "ood_rmse_ratio": 1.7215344384292102,
      "parameters": 347930
    },
    {
      "method": "I72",
      "id_mae": 9.377215514625831,
      "id_ratio": 1.1017023429130688,
      "eligible": false,
      "ood_mae_ratio": 1.9934415361584328,
      "ood_rmse_ratio": 2.241722063610777,
      "parameters": 347930
    },
    {
      "method": "I66",
      "id_mae": 8.720856423229998,
      "id_ratio": 1.0245885826869943,
      "eligible": true,
      "ood_mae_ratio": 2.0200457953576,
      "ood_rmse_ratio": 2.2979926427669124,
      "parameters": 347930
    },
    {
      "method": "I64",
      "id_mae": 8.542173789211924,
      "id_ratio": 1.0035956689346601,
      "eligible": true,
      "ood_mae_ratio": 2.24652802838496,
      "ood_rmse_ratio": 2.6608024604095317,
      "parameters": 347930
    },
    {
      "method": "I67",
      "id_mae": 9.029045305277435,
      "id_ratio": 1.0607968166645376,
      "eligible": false,
      "ood_mae_ratio": 2.2878821007909966,
      "ood_rmse_ratio": 2.703041659579243,
      "parameters": 347930
    },
    {
      "method": "I70",
      "id_mae": 12.09317907348908,
      "id_ratio": 1.4207931659190232,
      "eligible": false,
      "ood_mae_ratio": 2.3821642448488625,
      "ood_rmse_ratio": 2.5860713736191783,
      "parameters": 347930
    },
    {
      "method": "I71",
      "id_mae": 9.313336739186045,
      "id_ratio": 1.0941974075242358,
      "eligible": false,
      "ood_mae_ratio": 2.455968826272048,
      "ood_rmse_ratio": 2.9494552511085317,
      "parameters": 347930
    },
    {
      "method": "I68",
      "id_mae": 9.58636610388297,
      "id_ratio": 1.1262748499485364,
      "eligible": false,
      "ood_mae_ratio": 2.573825929872985,
      "ood_rmse_ratio": 2.818632717109793,
      "parameters": 347930
    },
    {
      "method": "I75",
      "id_mae": 11.80897709898687,
      "id_ratio": 1.3874030853901873,
      "eligible": false,
      "ood_mae_ratio": 2.724469002028643,
      "ood_rmse_ratio": 3.0767519011474733,
      "parameters": 347930
    },
    {
      "method": "I69",
      "id_mae": 9.06247620364917,
      "id_ratio": 1.0647245176973639,
      "eligible": false,
      "ood_mae_ratio": 3.478666082632854,
      "ood_rmse_ratio": 4.235781439963433,
      "parameters": 347930
    }
  ],
  "selection_split": "val",
  "reference_id_mae": 8.51156900495559,
  "id_limit": 8.766916075104257,
  "qualified_by_development_id": true,
  "no_extra_training": true,
  "evidence_sha256": {
    "I64": "e241fdd622bdfc50b597180e1fc0ac59bbf3c6b54daa0bf25eab1a4db4176c05",
    "I65": "7195377f20d492f618be752316071ac2739e15ad024bc8c2af7b5ae0b284159f",
    "I66": "a6f782d7970e3a2ac324be53fece97f9f12560b73518406b29bacacd78571f77",
    "I67": "39d8425db17f773d7aab80db050d769ec9f66cbb60afa26550289d4a0509284c",
    "I68": "9bfe9f41edd8f3e591450cc0c811e01eb977a9aeea0cff8c17aa155753e8fb06",
    "I69": "aea986e190a0b672bc3e40a1248f51ab2198c20dd680b4934643eff4c65d46d0",
    "I70": "15c8af1b1c17b95fb45c1c4d8ab852b45a36e5bb7d33c94107d8d2f66c0e57bf",
    "I71": "04a21e6c78a30f67d6345dfe3f32e337cc220776af967d97128b9e4deb822b03",
    "I72": "0dc41479a8201959405e72f98cd3655fa359c1def8301e7e593cbabc7e49da83",
    "I73": "536ba2943aa6d669da8f7be075eaec6d4c4440dfd00ff1d311d71a6590331024",
    "I74": "9c3bb406736656eebc65301c31e56fe086a7e95393135fb08f7af998678bf85e",
    "I75": "b24451c5b33a40e10c43fd452924093f161b6108e34333cfa1ae448ae2b01fd4"
  },
  "reference_sha256": "2b9ef11e982d90dd16b5b99b9a3a312f1ab2a46ff30e8b54368c5f4d4e9fdcfe"
}
```

### 复核记录与综合图

已核验12组各100轮、24份best/last可读且参数有限、三组评估目录各15模型×6协议，共270套评估；无缺组。552份冻结文件摘要、4份数据摘要、参考best与初始化摘要一致；12组公共初始化摘要相同。所有best均与日志中的最低ID验证MAE及epoch一致，last含优化器、调度器、AMP、RNG和训练状态。队列状态finished，启动器退出码0。未重新训练或执行GPU测试。

同一评估组内各模型的mask_spec、effective_mask_seed、样本数、缺失计数、受约束与无约束计数逐项一致；开发选择证据摘要与selection.json一致，筛选使用val。原I37/I56/I57是已有训练产物，只重做评估，没有重复训练。

核验记录：[verification.json](audits/verification.json)；可复算摘要：[reviewed_summary.json](audits/reviewed_summary.json)；逐组训练复核：[training_review.json](audits/training_review.json)。

十二组累计训练16.938小时，不含额外评估。best分布为90轮3组、95轮3组、100轮6组；固定100轮预算下的比较不代表各结构的永久性能上限。训练GPU日志均含start/sample/stop。日志可作为运行线索，不能替代内核硬件事件日志。

![ID与OOD折中](figures/id_ood_tradeoff.png)

左图为ID MAE相对原I37变化，右图为三类别等权相对OOD MAE变化；越低越好。右图中的3%改善线为预定要求。

### 路径及无约束区域实测

|组|ID主尺度路径/占比|几何缺失主路径/占比|几何缺失raw MAE|post MAE|无粗尺度约束区域MAE|ID网格等价专家成本|
|---|---|---|---:|---:|---:|---:|
|I37|MMFF 100.0%|MMFF 100.0%|13.49171|10.82116|18.97256|5.000|
|I57|FMFF 99.0%, FMCF 0.6%|FMFF 98.3%, FMMF 1.3%|15.26688|12.99449|25.55563|6.483|
|I64|CMFF 49.1%, FMCF 47.9%|FMCF 55.0%, CMFF 27.9%|28.12739|22.04208|43.74920|4.575|
|I65|FCFF 79.1%, CCFF 20.9%|FCFF 89.4%, CCFF 10.6%|19.11085|14.84953|29.43981|5.732|
|I66|CFFF 81.9%, CMFF 17.4%|CFFF 76.2%, CMFF 22.6%|25.92673|19.61221|39.33309|5.843|
|I69|MFCF 100.0%|MFCF 100.0%|30.52639|24.19962|38.03555|4.625|
|I70|FMCF 99.7%, CMCF 0.3%|FMCF 100.0%|26.74475|20.10284|37.93553|4.620|
|I71|FMCF 100.0%|FMCF 100.0%|25.60218|19.80391|34.38511|4.625|
|I72|FMFF 100.0%|FMFF 100.0%|26.94714|20.88081|41.94500|6.500|
|I73|CMCF 100.0%|CMCF 100.0%|22.10267|18.02926|33.76911|2.750|
|I74|MCFF 97.5%, MFFF 2.5%|MCFF 98.4%, MFFF 1.6%|19.02062|15.34677|26.90607|4.673|
|I75|MFFF 50.1%, MFMF 49.9%|MFFF 74.7%, MFMF 25.3%|30.65111|23.28613|40.92006|5.751|

上表几何缺失采用第一套mask；主结果中的“形态MAE”还平均第二套mask，因此数值略有差别。这里无粗尺度约束区域均有1,486,100个缺失元素。网格等价成本考虑尺度面积；每窗口仍为8次原生专家调用，不是这里显示的网格等价数。

### 预测输入系数与实际幅度

|组|配置预测gain|ID末轮实际P0注入绝对幅度|除以gain后的初始P0绝对幅度|
|---|---:|---:|---:|
|I65|1|20.63228|20.63228|
|I70|1|83.46958|83.46958|
|I72|0.25|45.30764|181.23055|
|I73|0.5|70.38203|140.76406|
|I75|0.5|3.15845|6.31690|

第四轮固定F，F缺失槽没有可见F聚合贡献，该诊断因此对应预测补位项。各组是独立训练后的不同模型，不是同一模型的评估时系数干预。该表不能用来推断单个参数的因果响应。

## 3. 分析、证据边界与建议

**本轮没有找到综合优于原I37的配置，十二组都未达到预定ID/OOD综合推荐标准。保留I37作为当前seed7候选，不用本轮配置替换。**

### 3.1 I65有ID收益，但没有获得泛化收益

I65只将逐轮专家输入的位置特征置零，H0、路由元数据、原生mask和support仍保留。ID MAE从8.05104降至7.88539，改善2.06%；ID RMSE从14.66026降至14.06841。但二元、形态、三元MAE分别从8.2973/10.8336/7.8921升至11.7011/14.8353/10.7034，类别相对比平均退步37.86%，最差类退步41.02%。

开发集在ID合格候选中选中I65，只说明它是本批新组中相对较好的选择；它的开发OOD仍比I37差37.57%，不是开发协议上的整体优化成功。新mask确认结果仍是ID改善1.16%、OOD退步38.76%，最差类退步42.37%，不支持把原测试退步归因于某一次mask抽样。

### 3.2 不支持“重复辅助输入整体有害”

原I37的support和position都开时，ID/OOD三类均值为8.0510/9.0076；只去support的I64为8.0826/20.0997；只去position的I65为7.8854/12.4133；都去掉的I66为8.1537/18.0449。删除位置具有ID优势，但三种删除都损害OOD。不能把“输入更少”直接当作“模型更可靠”。

在整块fallback清零且路由改为观测统计的另一输入条件下，已有I56两类元数据全开为10.2967/16.8841，I57两类全关为8.3992/10.1616；只恢复support的I67为8.5268/20.2129，只恢复position的I68为9.0133/22.3425。I57的清理收益明显依赖其他输入条件，单独恢复一种辅助特征并未取回收益。它们属于同预算独立训练的结构控制，仍不足以推广成某一类特征普遍有害。

### 3.3 补位与路由存在明显的条件依赖

在逐轮support/position都关时，原补位＋原路由(I66)、零补位＋原路由(I69)、原补位＋观测路由(I70)、零补位＋观测路由(I57)的ID/OOD均值依次为8.1537/18.0449、8.4325/29.6106、11.2788/20.7608、8.3992/10.1616。仅清理一路未带来独立改善；同时清理两路在这一条件下反而更好。这说明信息路径需要共同解释，不能据单个对照删除路径。

该表显示相互作用，但不能直接证明某种严格的“冗余机制”。不同配置会训练出不同参数、尺度路径和专家选择，结构改动的总体效应包含这些变化。

### 3.4 只保留可见F不自动改善，预测剂量结果也不单调

I71修复了整块清零同时删除已知F贡献的问题，实现语义更清楚，但相对I57，ID退步2.59%、OOD退步113.66%。这不说明已知F观测有害，而是表明将“已知部分总量/完整区域面积”作为缺失M/C槽的部分密度输入，没有在当前网络中产生更好的结果。该值仍不是完整粗尺度观测，mask维持缺失，且完整F/M/C早已进入H0。

在保留可见F、关闭重复元数据和清理路由的固定条件下，预测gain=0/0.25/0.5/1对应I71/I72/I73/I70，ID为8.6165/8.7059/8.5142/11.2788，OOD均值为20.8624/17.9161/15.1503/20.7608。0.5比0和1好，但没有优于I37或I57，不能把0.5选成已验证的最终强度。

**额外诊断：固定小系数不等于实际注入一定小。** I72(g=0.25)实际记录的F缺失槽P0补位幅度为45.31，I73(g=0.5)为70.38，而原强度I65(g=1)只有20.63。由于P0和相关网络共同学习，缩小系数的模型可以产生更大的原始P0。本批仅限定了一条路径的乘数，没有提供注入幅度上界。模型是否出现了具体的参数补偿，仍需对应参数/预测分析；现有结果已足以否定“系数更小，所以跨模型输入幅度必然更小”的说法。

### 3.5 弱预测与弱元数据没有可加收益

I74在I71上加0.25倍support/position，相对I71的OOD改善36.52%，但ID退步3.52%；相对I37，ID/OOD仍退步10.80%/46.73%。I73相对I71也改善部分结果，两者预定组合I75却得到ID=10.9844、OOD均值=23.4351；相对I73的ID/OOD退步29.01%/57.25%，相对I74退步23.14%/80.27%。

因此这轮明确不支持“各自局部有益，组合后就能吸收优点”。各输入还会改变状态分布、路由路径和专家训练，不能按孤立指标简单相加。这里是合理机制假设，不是已经证实的唯一原因。

### 3.6 退步主要已存在于投影前，不能仅归咎于硬约束或尺度路径

第一套几何缺失中，I37投影前/后MAE为13.4917/10.8212，无约束区域18.9726；I65为19.1109/14.8495、无约束29.4398；I75为30.6511/23.2861、无约束40.9201。投影在这些例子中仍降低整体误差，投影后的可见M/C总量残差保持约1e-5。骨干预测在投影前已经变差，约束无法补充没有观测覆盖区域的真实细节。

路径也改变了：I37的seed7为MMFF，I65主要FCFF/CCFF，I71主要FMCF，I73为CMCF，I74主要MCFF。但I57和I72都主要FMFF，OOD均值仍分别为10.1616与17.9161；I70与I71都主要FMCF，ID也有明显差别。因此尺度路径可以是影响因素，不能把所有失败单独归因于“选错路径”。路径固定/干预实验才可进一步识别其因果作用；本轮不额外启动训练。

### 3.7 最终选择及写作边界

1. 保留原I37作为当前候选；I65可保留为ID收益与OOD代价的消融证据，I57可保留为条件性输入清理参考。本轮没有整体更优的新候选。
2. 本轮是seed7、TaxiBJ、固定100轮预算的结论；不能用它推断全体训练种子，也不能据此声称原I37全局最优。先前其他种子结果仅作背景。
3. 模型有真实多尺度输入；前三轮自由尺度、末轮固定F，与固定CMFF要区分。本批并没有改历史池、差分或专家协作算法，不能用输入控制结果证明那些模块的独立必要性。
4. 单个尺度路径、输入幅度和粗尺度总量一致性都不足以保证缺失F误差逐轮下降。保留完整负结果、RMSE、确认mask及有效计数。
5. 按“最后一轮”的预定边界结束组合搜索，不新增第十三组，不额外进行训练种子搜索。后续优先整理固定模型的数据流、已有公平消融及论文表述；实验尚未支持的因果解释保持为假设。
