#!/usr/bin/env python3
import copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SIZES=[(8,8,'fixed'),(12,12,'fixed'),(16,16,'fixed'),(20,20,'fixed'),(24,24,'fixed'),(40,40,'fixed'),(16,16,'linear_in'),(16,16,'linear_out'),(16,16,'linear_both'),(24,24,'linear_both')]
def main():
 folder=ROOT/'configs/latent_size';folder.mkdir(exist_ok=True);base=json.loads((ROOT/'configs/i37_performance.json').read_text());rows=[]
 for i,(h,w,projection) in enumerate(SIZES,1):
  name=f'GS{i:02}';cfg=copy.deepcopy(base);cfg['model']['architecture']='v25_spatial_size_coe'
  cfg['model']['coe']['latent_size']={'lh':h,'lw':w,'projection':projection}
  (folder/f'{name}.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
  rows.append(dict(name=name,control='I37 (32×32)' if i<=6 else ('GS05 (24×24)' if i==10 else 'GS03 (16×16)'),description=f'隐藏网格{h}×{w}；空间映射{projection}；通道64、时间12及其他模块固定',config=f'configs/latent_size/{name}.json',spec={'lh':h,'lw':w,'projection':projection}))
 (folder/'suite.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
