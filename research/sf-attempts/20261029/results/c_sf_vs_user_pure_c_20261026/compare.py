"""Compare user-supplied pure-C curves with SF at equal step and total bpp.

A change in the pre-existing 10K gap is descriptive, not a causal estimator.
"""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

OUT = Path(__file__).resolve().parent
CSV = Path('/home/ls/.codex/attachments/9c35c1ef-9239-4a1b-8838-a7fe0be8e577/all_quality_points.csv')
RUN = Path('/data2/ls/checkpoints/c_quant_residual_sf_stable_20261026')
METRICS = ('psnr', 'lpips', 'dists', 'ms_ssim')
RATES = (.02, .04, .06, .08)
pure = {}
for row in csv.DictReader(CSV.open(newline='')):
    pure.setdefault(int(row['step']), []).append({k:float(v) for k,v in row.items() if k != 'step'})
for step,points in pure.items():
    assert len(points)==6 and {p['q'] for p in points}=={0.,.2,.4,.6,.8,1.}
    assert np.isfinite([[p[k] for k in ('actual_bpp',*METRICS,'validation_loss')] for p in points]).all()
assert len(pure)==47 and sorted(pure)==list(range(2000,94001,2000))
sf={};sources=[CSV];identity=None
for p in sorted((RUN/'joint').glob('validation_*.json')):
    step=int(p.stem.split('_')[-1])
    if step not in pure: continue
    report=json.loads(p.read_text())
    if identity is None: identity=report['_validation_identity']
    assert report['_validation_identity']==identity
    assert identity['images']==24 and identity['crop']==0 and identity['actual_rate'] is True
    sources.append(p)
    sf[step]=[dict(q=float(q),actual_bpp=r['rate_actual'],validation_loss=r['loss'],**{m:r[m] for m in METRICS})
              for q,r in report.items() if not q.startswith('_')]
    assert len(sf[step])==6
STEPS=sorted(sf);assert 10000 in sf
LATEST=max(sf)
def interpolate(points,metric,rate):
    points=sorted(points,key=lambda p:p['actual_bpp']);x=np.array([p['actual_bpp'] for p in points])
    assert np.all(np.diff(x)>0) and x[0]<=rate<=x[-1]
    return float(PchipInterpolator(np.log(x),[p[metric] for p in points],extrapolate=False)(np.log(rate)))

matched={}
for step in STEPS:
    matched[str(step)]={}
    for rate in RATES:
        c={m:interpolate(pure[step],m,rate) for m in METRICS}
        s={m:interpolate(sf[step],m,rate) for m in METRICS}
        raw={m:s[m]-c[m] for m in METRICS}
        prior={m:interpolate(sf[10000],m,rate)-interpolate(pure[10000],m,rate) for m in METRICS}
        change={m:raw[m]-prior[m] for m in METRICS}
        # Positive consistently means the SF run improved more since 10K.
        extra={m:change[m]*(1 if m in ('psnr','ms_ssim') else -1) for m in METRICS}
        matched[str(step)][str(rate)]=dict(pure_c=c,c_sf=s,sf_minus_pure_c=raw,
                                         gap_change_since10k=change,
                                         extra_improvement_since10k_positive_is_better=extra)
q_deltas={}
for s in STEPS:
    q_deltas[str(s)]={str(p['q']):{m:next(r for r in sf[s] if r['q']==p['q'])[m]-p[m]
                                 for m in ('actual_bpp',*METRICS)} for p in pure[s]}
limits=[
    'User confirmed identical 50K latent warmup, 4/1/1 + 0.1 base objective and Kodak24 EMA evaluation.',
    'CSV contains no dataset fingerprint, commit, seed, optimizer/batch details or checkpoint hash; these identities cannot be independently audited from the supplied file.',
    'The two runs already differ at 10K before SF activation. Equal named configuration does not establish an identical pre-SF state.',
    'Raw 18K gaps include the pre-existing 10K gap. Change in that gap is a descriptive comparison of learning trajectories, not proof of a causal SF effect.',
    'Matched rates use PCHIP on log actual total packet bpp within measured range, without extrapolation. They are interpolated values, not new model inference.',
    'No per-image metrics or repeated seeds are present, so no paired uncertainty interval or significance claim is available.',
    'The 94K pure-C curve is contextual only; the main comparison uses the same validation step for both runs.',
    'Training has not been modified or stopped for this analysis.'
]
report=dict(created_at=datetime.now(timezone.utc).isoformat(),latest_common_step=LATEST,common_steps=STEPS,
            csv_steps=dict(count=len(pure),minimum=min(pure),maximum=max(pure)),validation_identity_sf=identity,
            user_confirmed_same_configuration=True,sf_first_update=10001,matched_rate=matched,
            same_q_sf_minus_pure=q_deltas,
            mean_validation_rd={str(s):dict(pure_c=float(np.mean([p['validation_loss'] for p in pure[s]])),
                                              c_sf=float(np.mean([p['validation_loss'] for p in sf[s]]))) for s in STEPS},
            pure_c_94000_matched={str(r):{m:interpolate(pure[max(pure)],m,r) for m in METRICS} for r in RATES},
            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},limits=limits)
(OUT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
with (OUT/'matched_rate.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=('step','bpp','model',*METRICS));writer.writeheader()
    for step,byrate in matched.items():
        for rate,rows in byrate.items():
            for key in ('pure_c','c_sf'):writer.writerow(dict(step=step,bpp=rate,model=key,**rows[key]))
with (OUT/'additional_improvement.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=('step','bpp',*METRICS));writer.writeheader()
    for step,byrate in matched.items():
        for rate,row in byrate.items():
            writer.writerow(dict(step=step,bpp=rate,**row['extra_improvement_since10k_positive_is_better']))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(13.4,7.5))
low=max(min(p['actual_bpp'] for p in source) for source in (pure[10000],sf[10000],pure[LATEST],sf[LATEST]))
high=min(max(p['actual_bpp'] for p in source) for source in (pure[10000],sf[10000],pure[LATEST],sf[LATEST]))
grid=np.linspace(low,high,240)
for ax,m in zip(axes[0],METRICS[:3]):
    for name,points,color,style in ((f'Pure C / {LATEST//1000}K',pure[LATEST],'#377eb8','-'),
                                    (f'C+SF / {LATEST//1000}K',sf[LATEST],'#c52b39','-'),
                                    ('Pure C / 10K',pure[10000],'#377eb8','--'),
                                    ('SF run before SF / 10K',sf[10000],'#c52b39','--')):
        ax.plot(grid,[interpolate(points,m,r) for r in grid],label=name,color=color,linestyle=style,linewidth=1.9 if style=='-' else 1.2)
    ax.set(xlabel='Actual total packet bpp',ylabel={'psnr':'PSNR (dB), higher better','lpips':'LPIPS, lower better','dists':'DISTS, lower better'}[m],xlim=(low,high))
    ax.grid(alpha=.2)
for ax,m in zip(axes[1],METRICS[:3]):
    for rate,color in zip(RATES,('#7662a7','#dc9933','#277b59','#377eb8')):
        ys=[matched[str(s)][str(rate)]['extra_improvement_since10k_positive_is_better'][m] for s in STEPS]
        ax.plot(np.array(STEPS)/1000,ys,label=f'{rate:.2f} bpp',marker='o',markersize=3,color=color)
    ax.axhline(0,color='#333333',linestyle=':',linewidth=1)
    ax.set(xlabel='Training steps (K)',ylabel=f'Extra {m.upper()} improvement since 10K')
    ax.grid(alpha=.2)
fig.suptitle(f'Pure C vs C+SF | equal-step Kodak24 EMA at {LATEST//1000}K',fontsize=13)
fig.text(.5,.93,'Top: actual curves. Bottom: difference in improvement from each run\'s own 10K state; positive favors SF.',ha='center',fontsize=9)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.91),ncol=4,frameon=False,fontsize=9)
axes[1,2].legend(loc='best',fontsize=8,frameon=False)
fig.subplots_adjust(left=.07,right=.985,top=.83,bottom=.08,hspace=.38,wspace=.3)
fig.savefig(OUT/'pure_c_vs_sf.png',dpi=170);fig.savefig(OUT/'pure_c_vs_sf.svg');plt.close(fig)
lines=['# 纯 C 与 C+SF：相同步数、相同实际总码率','',f'纯 C CSV 共 47 个 checkpoint，2K–94K；两轮最新共同验证步数为 {LATEST}。用户确认 warmup、损失和 Kodak24 EMA 口径相同。','',
       f'## {LATEST//1000}K 的直接性能比较','', '| 总 bpp | 模型 | PSNR ↑ | LPIPS ↓ | DISTS ↓ |','|---:|---|---:|---:|---:|']
for rate,rows in matched[str(LATEST)].items():
    for key,label in (('pure_c','纯 C'),('c_sf','C+SF')):
        p=rows[key];lines.append(f"| {rate} | {label} | {p['psnr']:.4f} | {p['lpips']:.5f} | {p['dists']:.5f} |")
lines+=['','## 相对各自 10K 起点的额外改善','',
        '计算方式：C+SF 的 10K→当前改善量，减去纯 C 的同期改善量。正值表示 C+SF 进步更多；它只是轨迹比较，不是严格因果估计。','',
        '| 总 bpp | PSNR 额外改善 (dB) | LPIPS 额外降低 | DISTS 额外降低 |','|---:|---:|---:|---:|']
for rate,row in matched[str(LATEST)].items():
    e=row['extra_improvement_since10k_positive_is_better'];lines.append(f"| {rate} | {e['psnr']:+.5f} | {e['lpips']:+.6f} | {e['dists']:+.6f} |")
lines+=['','## 判断','',
        'C+SF 在 18K 的绝对指标更好，但未开启 SF 的 10K 已有优势。扣除原有差距后，PSNR 没有新增收益，LPIPS 仅有局部小幅信号，DISTS 方向混合。当前证据不足以支持 SF 带来明确、稳定的额外压缩收益。',
        '最有判别力的后续对照是从本轮相同 pre_sf.pt 分叉，保持相同数据顺序、batch、优化器、随机性、数值精度和更新预算，单独改变 SF；当前分析不改动正在运行的实验。','',
        '## 比较边界','',*['- '+s for s in limits]]
(OUT/'report.zh.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(output=str(OUT),latest_common_step=LATEST,latest_matched=matched[str(LATEST)],mean_validation_rd=report['mean_validation_rd']),indent=2))
