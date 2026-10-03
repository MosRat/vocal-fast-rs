from __future__ import annotations
import argparse, json, struct
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--vin',type=Path,required=True); ap.add_argument('--debug',type=Path,required=True); a=ap.parse_args()
    raw=a.vin.read_bytes(); frames,width,latent,up,plen,sid=struct.unpack_from('<6I',raw,8); cur=32
    def take(dt,n):
        nonlocal cur
        x=np.frombuffer(raw,dtype=dt,count=n,offset=cur).copy(); cur += x.nbytes; return x
    feats=take('<f4',frames*width).reshape(1,frames,width); take('<i4',2); rnd=take('<f4',latent*frames); pitch=take('<i4',frames); pitchf=take('<f4',frames); noise=take('<f4',frames*up)
    d=json.loads(a.debug.read_text()); pf=np.asarray(d['rvc_features'],dtype='<f4').reshape(1,frames,width); pp=np.asarray(d['pitch'],dtype='<i4'); pff=np.asarray(d['aligned_pitchf'],dtype='<f4')
    def m(x,y):
        z=y.astype(np.float64)-x.astype(np.float64); return {'max':float(np.max(np.abs(z))),'rmse':float(np.sqrt(np.mean(z*z))),'rel':float(np.linalg.norm(z)/max(np.linalg.norm(x),1e-12))}
    print(json.dumps({'feats':m(feats,pf),'pitch':m(pitch,pp),'pitchf':m(pitchf,pff),'rnd_head':rnd[:8].tolist(),'noise_head':noise[:8].tolist()},indent=2))
if __name__=='__main__': main()
