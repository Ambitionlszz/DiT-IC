import csv,json,math
from pathlib import Path
csvpath=Path('/home/ls/.codex/attachments/9c35c1ef-9239-4a1b-8838-a7fe0be8e577/all_quality_points.csv')
allrows=list(csv.DictReader(csvpath.open()))
root=Path('/data2/ls/checkpoints/c_context_sf_20261028/joint')
out=Path('/data2/ls/experiments/c_context_sf_20261028/comparisons')
metrics=['psnr','lpips','dists']
def lerp(rows,bpp,k):
 for a,b in zip(rows,rows[1:]):
  if a['bpp']<=bpp<=b['bpp']:
   t=(math.log(bpp)-math.log(a['bpp']))/(math.log(b['bpp'])-math.log(a['bpp']))
   return a[k]*(1-t)+b[k]*t
 raise ValueError(bpp)
available={int(p.stem.split("_")[-1]) for p in root.glob("validation_*.json")}
reference={int(r["step"]) for r in allrows}
for step in sorted(available & reference):
 v=json.loads((root/f'validation_{step:07d}.json').read_text())
 c=[dict(q=float(r['q']),bpp=float(r['actual_bpp']),rd=float(r['validation_loss']),**{m:float(r[m]) for m in metrics}) for r in allrows if int(r['step'])==step]
 s=[dict(q=float(q),bpp=d['rate_actual'],rd=d['loss'],**{m:d[m] for m in metrics}) for q,d in v.items() if not q.startswith('_')]
 c.sort(key=lambda r:r['bpp']);s.sort(key=lambda r:r['bpp'])
 same=[]
 for a,b in zip(c,s):
  same.append(dict(q=a['q'],C_bpp=a['bpp'],SF_bpp=b['bpp'],bpp_percent=100*(b['bpp']/a['bpp']-1),C_psnr=a['psnr'],SF_psnr=b['psnr'],**{'delta_'+m:b[m]-a[m] for m in metrics},rd_percent=100*(b['rd']/a['rd']-1)))
 lo=max(c[0]['bpp'],s[0]['bpp']);hi=min(c[-1]['bpp'],s[-1]['bpp'])
 points=[math.exp(math.log(lo)+i/100*(math.log(hi)-math.log(lo))) for i in range(101)]
 points[0],points[-1]=lo,hi
 deltas=[{m:lerp(s,x,m)-lerp(c,x,m) for m in metrics} for x in points]
 groups={'all':range(101),'low':range(34),'mid':range(34,67),'high':range(67,101)}
 summary=dict(step=step,method='linear interpolation in log(actual total bpp), 101 uniformly log-spaced common points, no extrapolation',
 overlap_bpp=[lo,hi],same_q=same,same_q_mean_bpp_percent=sum(r['bpp_percent'] for r in same)/6,
 same_q_mean_rd_percent=sum(r['rd_percent'] for r in same)/6,
 common_bpp_deltas={g:{m:sum(deltas[i][m] for i in ids)/len(ids) for m in metrics} for g,ids in groups.items()},
 packet_forward_max_error=max(d.get('packet_forward_max_error',0) for q,d in v.items() if not q.startswith('_')))
 (out/f'same_step_{step}_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary))
