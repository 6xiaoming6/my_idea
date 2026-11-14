#!/usr/bin/env python3
import copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SPECS=[
 ('LT01',{},'I37','128个无网格空间token/时间步，共享双向交叉注意力，直接写回'),
 ('LT02',{'count':64},'LT01','仅将每时间步token数量减为64'),
 ('LT03',{'count':256},'LT01','仅将每时间步token数量增为256'),
 ('LT04',{'mapping':'per_scale'},'LT01','F/M/C各自独立双向投影，初始参数复制相同，跨轮共享'),
 ('LT05',{'return_mode':'residual'},'LT01','将原生尺度状态增量写回token，保留输入token残差'),
 ('LT06',{'delta':False},'LT01','仅关闭token空间受控差分'),
 ('LT07',{'history':False},'LT01','仅关闭历史池读写'),
 ('LT08',{'delta':False,'history':False},'LT06/LT07','同时关闭差分与历史，仅顺序传递tokens'),
 ('LT09',{'path':'CMFF'},'LT01','仅固定尺度路径CMFF'),
 ('LT10',{'mapping':'per_scale','return_mode':'residual'},'LT04/LT05','预定组合：尺度独立投影与残差写回')]
def main():
 folder=ROOT/'configs/latent_tokens';folder.mkdir(exist_ok=True);base=json.loads((ROOT/'configs/i37_performance.json').read_text());rows=[]
 for name,change,control,description in SPECS:
  cfg=copy.deepcopy(base);cfg['model']['architecture']='v25_latent_token_coe'
  spec=dict(count=128,mapping='shared',return_mode='replace',delta=True,history=True,path='free_last_f');spec.update(change)
  cfg['model']['coe']['latent_tokens']=spec
  (folder/f'{name}.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
  rows.append(dict(name=name,control=control,description=description,config=f'configs/latent_tokens/{name}.json',spec=spec))
 (folder/'suite.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
