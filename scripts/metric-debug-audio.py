import argparse, json, numpy as np
ap=argparse.ArgumentParser(); ap.add_argument('--debug', required=True); ap.add_argument('--native', required=True); a=ap.parse_args()
x=np.asarray(json.load(open(a.debug))['audio'],dtype=np.float64); y=np.fromfile(a.native,dtype='<f4').astype(np.float64); d=y-x
print(json.dumps({'n':int(x.size),'max_abs':float(np.max(np.abs(d))),'rmse':float(np.sqrt(np.mean(d*d))),'relative_l2':float(np.linalg.norm(d)/max(np.linalg.norm(x),1e-12)),'cosine':float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))},indent=2))
