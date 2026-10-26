# I64–I75：最后一轮seed7输入优化

完整设计见 `model_designs/20261023_v25_I64-I75最后一轮seed7优化方案.md` 及同名JSON。固定十二组，全部seed7，复用原I37/I56/I57只读参考，不增加训练种子或测试后临时组合。

## 具体方法

I64/I65/I66分别仅删除逐轮专家输入support、position、两者；保留原专家预测补位与路由。
I67/I68在I57上分别恢复support、position；整块缺失补位仍为0，路由使用可见F统计。
I69/I70和I66、I57构成无重复metadata条件下的专家补位×路由统计对照。
I71保留可见F贡献，在原生M/C缺失槽输入pool(where(m_F,X_F,0))；I72/I73再向细尺度缺失槽加入0.25/0.5倍P0，然后进行同样空间平均。
I74在I71上加回0.25倍support和position；I75预先组合I73与I74。

`support_gain`、`position_gain`只缩放逐轮专家输入，不改H0及路由统计；原mask保持。
`expert_fill=visible_initial`时，`prediction_gain`只缩放细尺度缺失槽预测，再进行空间mean聚合；g=1严格等于原回填路径。可见F和原生M/C观测不缩放。g=0与I57的整块清零不同，前者保留已知F贡献。
例如2×2区域可见F为10/40、缺失P0为8/12，g=0/.25/.5/1的fallback为12.5/13.75/15/17.5。原生M/C可见时优先使用真实观测，派生值不改变mask。P0不detach、不刷新；不是新观测。

## 公共配置与冻结

TaxiBJ双通道12步，四族混合mask0.4，沿用2452/342/707窗口；H0真实F/M/C融合、四轮共享八专家，每轮原生Top2，前三轮自由尺度、第四轮F。原I37受控差分、历史池、学习分配投影保持。
100epoch、batch32、每5轮ID验证MAE选best；AdamW1e-3余弦至3e-4，wd1e-4，clip1，AMP，物理GPU1串行。损失：投影前/后缺失F各0.5，可见rawF0.1，可见M/C0.1，均衡0.01。公共未训练初始化，不加载已训练参考权重。
所有源码/配置/设计/数据/参考best/初始化摘要冻结；十二组均347930参数，没有裁剪置零通道，因此不宣称算力节省。旧配置默认行为保持。

## 验收、队列与评估

CPU验证旧中性配置前向/损失/梯度严格一致，新十二组同初始化与H0、手算fallback及梯度、可见证据不缩放、metadata独立缩放、g=1原路径精确一致、隐藏污染不影响输出、sum/mean一致、空观测有限、8次专家调用及投影约束。
GPU1预检每组原尺寸batch32 AMP原生及强制TA/ST三次有效更新，并检查完整checkpoint恢复。通过后按I64→I75训练。
每组完成立即跑验证窗口六套开发协议（mask基种子2026102301–06，保留split偏移）。十二组完成后只用开发结果排序，ID≤I37同协议103%，OOD三类等权类别相对MAE比值，同分按OOD RMSE/ID/参数/编号。无合格组则不强选新方法。
冻结选择记录后，所有十二组及三参考跑原六套测试和新确认mask2026102311–16。二元/形态各先平均两次mask，三元为一套；不把mask复测当训练种子。最终推荐要求ID代价≤3%、OOD改善≥3%、最差类别退步≤5%，不能宣称显著性或全局最优。
训练保留best/last、完整恢复状态、每15秒`logs/gpu_status.jsonl`与原诊断；队列失败停止、完成跳过，中断从last恢复。tmux主进程训练/评估/正式报告完成后自动退出，baseline不恢复。正式报告按项目约定写入`experments_report/`。

```bash
python -m unittest discover -s tests -p test_v25_final_input.py -v
python scripts/v25/run_final_input.py --freeze
CUDA_VISIBLE_DEVICES=1 python -u <suite>/source_snapshot/scripts/v25/preflight_taxibj_input_communication.py --suite <suite>
python -u <suite>/source_snapshot/scripts/v25/run_final_input.py --suite <suite>
```
