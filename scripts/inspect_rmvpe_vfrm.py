import json, struct, sys
p=sys.argv[1]
b=open(p,'rb').read()
m,c,d,o=struct.unpack_from('<QQQQ', b, 16)
cfg=json.loads(b[64:64+m])
print(json.dumps(cfg.get('metadata',{}), indent=2))
cur=64+m
for i in range(c):
    n,dt,rank,_=struct.unpack_from('<IBBH',b,cur);cur+=8
    name=b[cur:cur+n].decode();cur+=n
    shape=struct.unpack_from('<'+'Q'*rank,b,cur);cur+=8*rank
    off,l=struct.unpack_from('<QQ',b,cur);cur+=16
    if name.startswith('rmvpe.') or i<10:
        print(i, name, shape, dt, off, l)
