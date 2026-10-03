import argparse
import json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--a', required=True)
ap.add_argument('--b', required=True)
x = ap.parse_args()
a = np.fromfile(x.a, dtype='<f4')
b = np.fromfile(x.b, dtype='<f4')
d = b - a
print(json.dumps({'n': int(a.size), 'max_abs': float(np.max(np.abs(d))),
                  'rmse': float(np.sqrt(np.mean(d*d))),
                  'relative_l2': float(np.linalg.norm(d) / max(np.linalg.norm(a), 1e-12)),
                  'cosine': float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))}, indent=2))
