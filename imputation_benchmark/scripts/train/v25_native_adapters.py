"""Data bridges only. Upstream model files remain byte-for-byte unchanged."""
from __future__ import annotations
import ast
import importlib
import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path
import torch
import numpy as np

METHODS = ['SAITS', 'GRIN', 'ImputeFormer', 'T1', 'PAST', 'STAMImputer', 'BRITS', 'MRNN', 'BiMPGRU', 'MPGRU', 'RITS']
REPOS = {'SAITS':'SAITS','BRITS':'SAITS','RITS':'SAITS','MRNN':'SAITS','GRIN':'grin',
         'BiMPGRU':'grin','MPGRU':'grin','T1':'T1_official','ImputeFormer':'ImputeFormer_official',
         'PAST':'PAST','STAMImputer':'STAMImupter'}
DESCRIPTIONS = {
 'SAITS':'原生双块自注意力；完整网格展开为2048特征；原生ORT+MIT训练。',
 'GRIN':'原生双向图循环补全；每通道完整1024节点四邻接网格；原生最终及中间预测L1。',
 'ImputeFormer':'原生低秩注意力，32维输入嵌入+96维可学习嵌入、3层；原生缺失L1+0.01 Fourier损失。',
 'T1':'作者PEMS03配置，128头、2+2块、卷积核71/31；原生缺失MSE；完整2048特征。',
 'BRITS':'SAITS仓库原生BRITS；完整2048特征、hidden64；原生重建与双向一致性损失。',
 'MRNN':'SAITS仓库原生MRNN；完整2048特征、hidden64；原生两路重建RMSE。',
 'BiMPGRU':'GRIN仓库原生双向消息传播GRU参考；完整四邻接网格、hidden64。',
 'MPGRU':'GRIN仓库原生单向消息传播GRU参考；完整四邻接网格、hidden64。',
 'RITS':'SAITS仓库原生单向RITS参考；完整2048特征、hidden64。',
 'PAST':'作者原生3层、hidden64图时序模型；真实日历特征；观测MSE及detach残差辅助MSE。',
 'STAMImputer':'作者训练入口MoE_v2：4层、4头、embedding32、dropout0.15；真实日历与原生db1小波；原生物理量非零缺失MAE。',
}

def package(name, path):
    """Load only required upstream modules, avoiding unrelated CLI dependencies."""
    if name not in sys.modules:
        m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m
    return sys.modules[name]

def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def load_tsl_leaf(root):
    # All implementations (including MLP, Dense and ops) are original vendored TSL files.
    base=root/'experiments/tsl'
    for name,rel in [('tsl',''),('tsl.nn','nn'),('tsl.nn.base','nn/base'),('tsl.nn.blocks','nn/blocks'),('tsl.nn.blocks.encoders','nn/blocks/encoders')]:
        package(name,base/rel)
    sys.modules['tsl'].epsilon=5e-8  # Same constant as upstream tsl/__init__.py.
    module=importlib.import_module('tsl.nn.blocks.encoders.mlp')
    sys.modules['tsl.nn.blocks.encoders'].MLP=module.MLP

def native_freg(root):
    # Execute the original Freg function AST without importing Lightning trainers.
    path=root/'experiments/imputeformer/imputers/imputeformer_imputer.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ImputeFormerImputer')
    func=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='Freg')
    import einops
    ns={'torch':torch,'rearrange':einops.rearrange}
    exec(compile(ast.Module(body=[func],type_ignores=[]),str(path),'exec'),ns)
    return ns['Freg']

def stam_functions(root):
    # The published script executes training on import; load its original pure functions only.
    import pywt
    path=root/'train_test.py';tree=ast.parse(path.read_text())
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['disentangle','masked_mae']]
    ns={'torch':torch,'np':np,'pywt':pywt}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(path),'exec'),ns)
    return ns

def adjacency(h=32,w=32):
    a=np.zeros((h*w,h*w),dtype=np.float32)
    for i in range(h):
        for j in range(w):
            for di,dj in [(0,1),(0,-1),(1,0),(-1,0)]:
                r,c=i+di,j+dj
                if 0<=r<h and 0<=c<w:a[i*w+j,r*w+c]=1
    return a

def build(name,roots,device,shape=(2,12,32,32)):
    c,t,h,w=shape;n=c*h*w;root=roots/REPOS[name]
    sys.path.insert(0,str(root))
    if name in ['SAITS','BRITS','RITS','MRNN']:
        if name=='SAITS':
            cls=importlib.import_module('modeling.saits').SAITS
            model=cls(n_groups=2,n_group_inner_layers=1,d_time=t,d_feature=n,d_model=256,d_inner=128,
                n_head=4,d_k=64,d_v=64,dropout=.1,input_with_mask=True,param_sharing_strategy='inner_group',
                MIT=True,device=str(device),diagonal_attention_mask=True)
        else:
            mod=importlib.import_module('modeling.mrnn' if name=='MRNN' else 'modeling.brits')
            model=getattr(mod,name)(seq_len=t,feature_num=n,rnn_hidden_size=64,device=str(device),MIT=False)
    elif name in ['GRIN','MPGRU','BiMPGRU']:
        mod=importlib.import_module('lib.nn.models.grin' if name=='GRIN' else 'lib.nn.models.mpgru')
        cls=getattr(mod,{'GRIN':'GRINet','MPGRU':'MPGRUNet','BiMPGRU':'BiMPGRUNet'}[name])
        kw=dict(adj=adjacency(h,w),d_in=1,d_hidden=64,d_ff=64,n_layers=1,kernel_size=2)
        if name=='GRIN':kw.update(ff_dropout=0.,d_emb=8)
        if name=='BiMPGRU':kw.update(embedding_size=8)
        model=cls(**kw)
    elif name=='ImputeFormer':
        load_tsl_leaf(root);sys.path.insert(0,str(root/'experiments'))
        cls=importlib.import_module('imputeformer.models').ImputeFormerModel
        model=cls(num_nodes=h*w,input_dim=1,output_dim=1,input_embedding_dim=32,learnable_embedding_dim=96,
                  feed_forward_dim=256,num_temporal_heads=4,num_layers=3,dropout=.1,windows=t,dim_proj=8)
        model._native_freg=native_freg(root)
    elif name=='T1':
        cls=load_file('t1_native',root/'models/T1.py').Model
        cfg=Namespace(task_name='imputation',seq_len=t,pred_len=t,label_len=0,enc_in=n,
            n_heads=128,patch_size=2,patch_stride=1,n_blocks=[2,2],kernel_size_large=[71,31],kernel_size_small=5,
            ffn_ratio=1.,downsample_ratio=2,positional_encoding=True,qkv_bias=True,cosine_attention=False,
            use_head_reconstruction=True,head_params_shared=True,imputation_use_mask_embedding=False,
            drop_attn=.1,drop_ffn=0.,drop_proj=0.,drop_path=.1,drop_head=0.,SACA=False,use_norm=1)
        model=cls(cfg)
    elif name=='PAST':
        package('past_native',root/'model')
        cls=importlib.import_module('past_native.model').Model
        cfg=Namespace(node_num=h*w,hidden_dim=64,dropout=.1,layer_num=3,seq_len=t,alpha=.1,order=1)
        model=cls(cfg,torch.as_tensor(adjacency(h,w),device=device))
    elif name=='STAMImputer':
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        package('model',root/'model')
        cls=importlib.import_module('model.MoE').SP_TSFormer_MoE_v2
        # Same nearest-neighbour construction as upstream loadGraph, using grid adjacency.
        adj=adjacency(h,w)+np.eye(h*w);dist=dijkstra(csgraph=csr_matrix(adj))
        dist[dist==0]=dist.max()+10;k=int(np.log2(h*w))
        neighbours=np.argpartition(dist,k,-1)[:,:k]
        model=cls(5,32,64,1,h*w,t,neighbours,4,4,.15,4)
        model.device=str(device)  # Device placement only; no structural change.
        model._native_helpers=stam_functions(root)
    else:raise ValueError(name)
    return model.to(device)

def flat(x):return x.permute(0,2,1,3,4).flatten(2)
def unflat(x,shape):
    b,c,t,h,w=shape;return x.reshape(b,t,c,h,w).permute(0,2,1,3,4)
def graph(x):
    b,c,t,h,w=x.shape;return x.reshape(b*c,t,h*w,1)
def ungraph(x,shape):return x.reshape(shape)
def gaps(m):
    d=torch.zeros_like(m)
    for t in range(1,m.shape[1]):d[:,t]=1+(1-m[:,t])*d[:,t-1]
    return d

def masked(pred,target,mask,power=1):
    e=(pred-target).abs()
    if power==2:e=e.square()
    return (e*mask).sum()/mask.sum().clamp_min(1)

def forward(name,model,batch,stats,device,training=False):
    # Sanitise unobserved slots. Coarse observations/ground truth never accessed.
    mask=batch['m_f'].to(device).bool()
    obs=torch.where(mask,batch['x_f_obs'].to(device),0.)
    mean=torch.as_tensor(stats['node'],device=device)[None,:,None]
    std=torch.as_tensor(stats['std'],device=device)[None,:,None]
    x=torch.where(mask,(obs-mean)/std,0.)
    target=(batch['x_f_gt'].to(device)-mean)/std if training else None
    loss=None
    if name in ['SAITS','BRITS','RITS','MRNN','T1']:
        v,m=flat(x),flat(mask).float()
        if name in ['BRITS','RITS','MRNN']:
            data={'forward':dict(X=v,missing_mask=m,deltas=gaps(m)),
                  'backward':dict(X=v.flip(1),missing_mask=m.flip(1),deltas=gaps(m.flip(1)))}
            out=model(data) if name=='RITS' else model(data,'train' if training else 'test')
            y=out['imputed_data']
            if training:loss=out['reconstruction_loss']+out.get('consistency_loss',0.)
        elif name=='SAITS':
            data=dict(X=v,missing_mask=m)
            if training:data.update(X_holdout=flat(target),indicating_mask=1-m)
            out=model(data,'train' if training else 'test');y=out['imputed_data']
            if training:loss=out['reconstruction_loss']+out['imputation_loss']
        else:
            y=model(v,mask=m)
            if training:loss=masked(y,flat(target),1-m,power=2)
        pred=unflat(y,x.shape)
    else:
        v,m=graph(x),graph(mask)
        if name=='ImputeFormer':
            y=model(v,None,m)
            if training:
                gt=graph(target)
                # All source training entries exist before our synthetic mixed corruption.
                # Native Freg(original_mask=ones) uses predictions everywhere, never hidden GT.
                loss=masked(y,gt,~m)+.01*model._native_freg(None,y.float(),gt,torch.ones_like(m))
        elif name=='PAST':
            cal=batch['calendar'].to(device).long().repeat_interleave(x.shape[1],0).unsqueeze(2)
            native_mask=m & (torch.rand(m.shape,device=device)>.1) if training else m
            main,aux=model(v,native_mask,cal);y=main+aux
            if training:loss=masked(main,v,m,2)+masked(main.detach()+aux,v,m,2)
        elif name=='STAMImputer':
            cal=batch['calendar'].to(device).repeat_interleave(x.shape[1],0)
            day=cal[:,:,1];slot=cal[:,:,2]*2+cal[:,:,3]//2
            time_features=torch.stack([day,slot],-1).unsqueeze(2).expand(-1,-1,v.shape[2],-1)
            arr=torch.cat([v,time_features],-1).detach().cpu().numpy()
            lo,hi=model._native_helpers['disentangle'](arr[:,:,:,:1],'db1',1)
            y=model(arr,lo,hi)
            if training:
                physical=ungraph(y,x.shape)*std+mean
                loss=model._native_helpers['masked_mae'](physical[~mask],batch['x_f_gt'].to(device)[~mask],0.)
        else:
            native_mask=m & (torch.rand(m.shape,device=device)>=.05) if training else m
            out=model(torch.where(native_mask,v,0.),native_mask)
            y=out[0] if isinstance(out,tuple) else out
            if training:
                loss=masked(y,v,m)
                if isinstance(out,tuple):
                    # GraphFiller.trim_seq unwraps the single prediction tensor,
                    # then its training_step iterates all four predictions.
                    for extra in out[1]:loss=loss+masked(extra,v,m)
        pred=ungraph(y,x.shape)
    pred=pred*std+mean
    return torch.where(mask,obs,pred),loss
