import copy,hashlib,json,sys,tarfile,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/cloud_gpu'),str(ROOT/'scripts/v24'),str(ROOT/'scripts/v25')]
import run as cloud
import unpack_data
from train_four_direction import train
from evaluate_four_direction import evaluate_sets
from stmoe_imputer.data.observed_multiscale import aggregate_sum
from stmoe_imputer.utils.deterministic import state_hash
from cloud_report import export_report

class CloudTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def jobs(self,methods):
        js=cloud.make_jobs('taxibj',methods,.4,epochs=2,batch=2,device='cpu')
        x=np.arange(4*2*3*8*8,dtype=np.float32).reshape(4,2,3,8,8)%51
        for split in ['train','val','test']:
            np.savez(self.root/(split+'.npz'),x_f_gt=x,x_m_gt=aggregate_sum(x,2),x_c_gt=aggregate_sum(x,4))
        for j in js.values():
            c=j['config'];c['model']['main']['dim']=8;c['model']['coe']['router_hidden_dim']=8
            c['data']['pin_memory']=False;c['train']['val_epoch']=1
            j['sources']={k:str(self.root/(k+'.npz')) for k in ['train','val','test']}
        return js

    def test_all_methods_shared_initialization_and_preflight(self):
        js=self.jobs(cloud.METHODS);cloud.initialize(self.root,js)
        cloud.write(self.root/'resolved_jobs.json',js);cloud.write(self.root/'plan.json',{'device':'cpu'})
        import cloud_preflight
        cloud_preflight.run(self.root)
        self.assertEqual(cloud.load(self.root/'acceptance/preflight.json')['status'],'passed')

    def test_history_resume_evaluate_report_and_tamper(self):
        js=self.jobs(['I03']);cloud.initialize(self.root,js);name,j=next(iter(js.items()))
        run=self.root/'runs'/name;receipt=self.root/'results'/(name+'.json')
        cloud.write(self.root/'plan.json',{'project_root':str(self.root)})
        cloud.write(self.root/'runs'/(name+'.json'),{'run_dir':str(run)})
        train(j,run,receipt,'cpu',stop_after=1)
        export_report(self.root,js);self.assertFalse(cloud.load(self.root/'report_path.json')['complete'])
        train(j,run,receipt,'cpu');train(j,self.root/'full',self.root/'full.json','cpu')
        a=torch.load(run/'checkpoints/last.pth',weights_only=False);b=torch.load(self.root/'full/checkpoints/last.pth',weights_only=False)
        for key in ['model','optimizer','scheduler','scaler','rng_states']:self.assertEqual(state_hash(a[key]),state_hash(b[key]),key)
        p=j['config']['experiment_plan']['protocol'];e=self.root/'evaluations'/(name+'.json')
        evaluate_sets(run/'checkpoints/best.pth',p,j['sources']['test'],e,cloud.digest(j['config']),'cpu')
        export_report(self.root,js);self.assertTrue(cloud.load(self.root/'report_path.json')['complete'])
        self.assertEqual(len(cloud.load(e)['sets']),6)
        bad=cloud.load(e);bad['config_sha256']='bad';cloud.write(e,bad)
        export_report(self.root,js);self.assertFalse(cloud.load(self.root/'report_path.json')['complete'])


    def test_frozen_subprocess_queue_and_skip_completed(self):
        import os,shutil,subprocess
        js=self.jobs(['I01','I02','I03']);suite=self.root/'suite';snap=suite/'source_snapshot'
        for name in ['src','scripts','configs']:
            shutil.copytree(ROOT/name,snap/name,ignore=shutil.ignore_patterns('__pycache__'))
        source={str(p.relative_to(snap)):unpack_data.sha(p) for p in snap.rglob('*') if p.is_file()}
        data={p:{'sha256':unpack_data.sha(p),'size':Path(p).stat().st_size} for p in next(iter(js.values()))['sources'].values()}
        cloud.write(suite/'plan.json',{'project_root':str(self.root),'jobs':js,'source':source,'data':data,'gpu':0,'device':'cpu','preflight':True})
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLCONFIGDIR=str(self.root/'mpl'))
        cmd=[sys.executable,str(ROOT/'scripts/cloud_gpu/run.py'),'--resume',str(suite)]
        with (self.root/'queue.log').open('w') as f:
            first=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        self.assertEqual(first.returncode,0,(self.root/'queue.log').read_text()[-5000:])
        self.assertEqual(cloud.load(suite/'completed.json')['count'],3)
        timestamps={p:p.stat().st_mtime_ns for p in (suite/'runs').rglob('*.pth')}
        with (self.root/'resume.log').open('w') as f:
            second=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        self.assertEqual(second.returncode,0,(self.root/'resume.log').read_text()[-5000:])
        self.assertEqual(timestamps,{p:p.stat().st_mtime_ns for p in timestamps})

    def test_dataset_install_rejects_changed_existing_bytes(self):
        source=self.root/'source';source.write_bytes(b'example')
        row={'target':'data/cloud_gpu/taxibj/train.npz','size':7,'sha256':unpack_data.sha(source)}
        (self.root/'datasets').mkdir();cloud.write(self.root/'datasets/manifest.json',{'datasets':{'taxibj':{'files':{'train':row}}}})
        archive=self.root/'data.tar'
        with tarfile.open(archive,'w') as t:t.add(source,arcname=row['target'])
        target=self.root/'installed'
        with patch.object(unpack_data,'ROOT',self.root):
            unpack_data.install(archive,target);unpack_data.install(None,target)
            (target/row['target']).write_bytes(b'changed')
            with self.assertRaises(ValueError):unpack_data.install(archive,target)

if __name__=='__main__':unittest.main()
