#!/usr/bin/env python3
"""Create the prespecified ten configurations; never modify them during a run."""
import copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SPECS=[
 ('L01',{},'I37','M网格基础版：统一16×16 latent，线性尺度映射、直接回写'),
 ('L02',{'factor':4},'L01','仅将latent缩小到C网格8×8，检验压缩代价'),
 ('L03',{'mapping':'learned'},'L01','双向尺度映射增加共享于各轮的零初始化残差1×1通道映射'),
 ('L04',{'mapping':'subcell'},'L01','双向映射改为空间子格重排与可学习通道压缩/展开'),
 ('L05',{'return_mode':'residual'},'L01','回写改为latent输入加映射后的原生尺度状态增量'),
 ('L06',{'delta':False},'L01','仅关闭受控差分，保留顺序状态与历史读取'),
 ('L07',{'history':False},'L01','仅关闭历史读取及写入，保留受控差分'),
 ('L08',{'delta':False,'history':False},'L06/L07','同时关闭差分与历史，只保留顺序latent传递'),
 ('L09',{'path':'CMFF'},'L01','仅将前三轮自由、末轮F改为固定CMFF'),
 ('L10',{'mapping':'learned','return_mode':'residual'},'L03/L05','预先确定的组合：可学习尺度映射与残差回写')]
def main():
 folder=ROOT/'configs/latent_space';folder.mkdir(exist_ok=True)
 base=json.loads((ROOT/'configs/i37_performance.json').read_text())
 rows=[]
 for name,change,control,description in SPECS:
  cfg=copy.deepcopy(base)
  cfg['model']['architecture']='v25_latent_space_coe'
  cfg['model']['coe']['latent_space']=dict(factor=2,mapping='linear',return_mode='replace',delta=True,history=True,path='free_last_f',**{})
  cfg['model']['coe']['latent_space'].update(change)
  (folder/f'{name}.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
  rows.append(dict(name=name,control=control,description=description,config=f'configs/latent_space/{name}.json',spec=cfg['model']['coe']['latent_space']))
 (folder/'suite.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
