import csv,json,math,statistics
from pathlib import Path

BASE=Path('/data2/ls/experiments/c_dual_rate_task_sf_20261027/comparisons')
C_PATH=Path('/home/ls/.codex/attachments/9c35c1ef-9239-4a1b-8838-a7fe0be8e577/all_quality_points.csv')
SF_ROOT=Path('/data2/ls/checkpoints/c_dual_rate_task_sf_20261027/joint')
METRICS=['psnr','lpips','dists','ms_ssim']
with C_PATH.open() as source:
    rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(source)]

def interp(points,x,metric):
    points=sorted(points,key=lambda p:p['bpp'])
    for a,b in zip(points,points[1:]):
        if a['bpp']-1e-12<=x<=b['bpp']+1e-12:
            t=(math.log(x)-math.log(a['bpp']))/(math.log(b['bpp'])-math.log(a['bpp']))
            return a[metric]+t*(b[metric]-a[metric])
    raise ValueError('no extrapolation')

def compare(step,sf_path):
    c=[dict(r,bpp=r['actual_bpp']) for r in rows if r['step']==step]
    assert len(c)==6
    data=json.loads(sf_path.read_text())
    sf=[dict(r,q=float(q),bpp=r['rate_actual']) for q,r in data.items() if not q.startswith('_')]
    same=[]
    for a in sorted(c,key=lambda r:r['q']):
        b=next(r for r in sf if r['q']==a['q'])
        same.append(dict(q=a['q'],C_bpp=a['bpp'],SF_bpp=b['bpp'],bpp_change_percent=(b['bpp']/a['bpp']-1)*100,
            **{prefix+m: row[m] for prefix,row in [('C_',a),('SF_',b)] for m in METRICS},
            **{'delta_'+m:b[m]-a[m] for m in METRICS},C_RD=a['validation_loss'],SF_RD=b['loss']))
    lo=max(min(r['bpp'] for r in c),min(r['bpp'] for r in sf))
    hi=min(max(r['bpp'] for r in c),max(r['bpp'] for r in sf))
    matched=[]
    for i in range(101):
        x=math.exp(math.log(lo)+(math.log(hi)-math.log(lo))*i/100)
        matched.append(dict(bpp=x,**{'delta_'+m:interp(sf,x,m)-interp(c,x,m) for m in METRICS}))
    c_rd=statistics.mean(r['C_RD'] for r in same);sf_rd=statistics.mean(r['SF_RD'] for r in same)
    summary=dict(step=step,source_C=str(C_PATH),source_SF=str(sf_path),same_q=same,
        same_q_mean_bpp_change_percent=statistics.mean(r['bpp_change_percent'] for r in same),
        same_q_bpp_change_range_percent=[min(r['bpp_change_percent'] for r in same),max(r['bpp_change_percent'] for r in same)],
        same_q_mean_delta={m:statistics.mean(r['delta_'+m] for r in same) for m in METRICS},
        C_mean_RD=c_rd,SF_mean_RD=sf_rd,aggregate_RD_change_percent=(sf_rd/c_rd-1)*100,
        matched_bpp=dict(method='Piecewise linear interpolation in log(total bpp), 101 log-spaced points inside measured overlap; no extrapolation',range=[lo,hi],
            mean_delta={m:statistics.mean(r['delta_'+m] for r in matched) for m in METRICS},
            low_third_mean_delta={m:statistics.mean(r['delta_'+m] for r in matched[:34]) for m in METRICS},
            high_third_mean_delta={m:statistics.mean(r['delta_'+m] for r in matched[-34:]) for m in METRICS}),
        limitations=['One run and Kodak24 mean curves; no statistical significance established.',
            'C BF16 versus SF FP16+GradScaler on V100; user confirmed same image set but directory layout differs.',
            'New SF also adds a zero-initialized quality conditioner and auxiliary image supervision; gains cannot be attributed to alignment alone.',
            'Use rate_actual on all24 images with side information; packet_bpp may cover only two decoded examples.'])
    return summary,matched

summaries=[]
for step in [12000,14000,16000,18000]:
    summary,matched=compare(step,SF_ROOT/f'validation_{step:07d}.json')
    (BASE/f'same_step_{step}_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for kind,data in [('same_q',summary['same_q']),('matched_total_bpp',matched)]:
        with (BASE/f'same_step_{step}_{kind}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    summaries.append({k:v for k,v in summary.items() if k not in ['same_q','limitations','source_C','source_SF']})
(BASE/'trend_12000_18000.json').write_text(json.dumps(summaries,indent=2)+'\n')
print(json.dumps(summaries,indent=2))
