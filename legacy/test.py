import sys; sys.path.insert(0,'.')
from sam3_relabel_video import load_cams
import glob, os, numpy as np
stems=[os.path.splitext(os.path.basename(f))[0] for f in sorted(glob.glob('data/replica_room0_v2/images/*.jpg'))]
c1=load_cams('data/replica_room0/sparse/0'); c2=load_cams('data/replica_room0_v2/sparse/0')
print('frames:',len(stems), ' non-v2 cover:',sum(s in c1 for s in stems)/len(stems), ' v2 cover:',sum(s in c2 for s in stems)/len(stems))
s=next(s for s in stems if s in c1 and s in c2)
print(s,' Δt:',np.linalg.norm(c1[s]['t']-c2[s]['t']),' ΔR max:',np.abs(c1[s]['R']-c2[s]['R']).max())
