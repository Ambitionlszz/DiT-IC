"""Pin current C/D validation reports and compare with audited 512 baselines."""
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import trapezoid

sys.path.insert(0, '/data2/ls/results/cd_tensorboard_20261021')
import dashboard as dashboard

OUT = Path(__file__).resolve().parent
ROOT = Path('/data2/ls/checkpoints/bt_self_flow_v1')
OLD = Path('/data2/ls/checkpoints/bt_self_flow_comparison_20261019')
METRICS = ('psnr', 'lpips', 'dists')
NOW = datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interpolate(curve, rate):
    x = np.array([row['avg_bpp'] for row in curve])
    if rate < x.min() or rate > x.max():
        return None
    return {metric: float(PchipInterpolator(np.log(x), [row[metric] for row in curve],
                 extrapolate=False)(np.log(rate))) for metric in METRICS}


def compare(reference, candidate):
    return {metric: dashboard.bd.bd_rate(reference, candidate, metric) for metric in METRICS}


def bd_text(result):
    return f'{result["percent"]:+.2f}%' if result['status'] == 'computed' else 'N/A'


OUT.mkdir(exist_ok=True, parents=True)
(OUT/'sources').mkdir(exist_ok=True)
runs = dashboard.read_logs()
history_reference = read(OLD/'historical_reference.json')
source_baseline = read(OLD/'comparison.json')
curves = {}
for key in ('baseline_30k', 'historical_200k'):
    curves[key] = [{name: point[name] for name in ('avg_bpp', *METRICS, 'num_images')}
                   for point in source_baseline['curves'][key]]
assert curves['historical_200k'] == [
    {name: point[name] for name in ('avg_bpp', *METRICS, 'num_images')}
    for point in history_reference['reference_curves']['A：非窗口训练 / 全局']]
for point, source in zip(curves['baseline_30k'], read(OLD/'baseline/all_summaries.json')['global']):
    assert all(point[key] == source[key] for key in point)
for filename in ('historical_reference.json', 'comparison.json'):
    shutil.copyfile(OLD/filename, OUT/'sources'/('baseline_'+filename))
old46 = read(Path('/data2/ls/results/cd_real_inference_20261020_46k/comparison.json'))
status = {}
latest = {}
sources = {}
for arm in ('C', 'D'):
    folder = ROOT/arm/'joint'
    train = runs[arm]['train'][max(runs[arm]['train'])]
    latest[arm] = max(runs[arm]['validation'])
    status[arm] = dict(train_step=train['step'], latest_validation_step=latest[arm],
        train_log=train, best=read(folder/'best.json'),
        validation_rd_mean=dashboard.mean_report(runs[arm]['validation'][latest[arm]])['loss'],
        validation_loss_history={str(step):dashboard.mean_report(value)['loss']
                                 for step,value in runs[arm]['validation'].items()},
        phase='joint', target_steps=200000)
common = max(set(runs['C']['validation']) & set(runs['D']['validation']))
candidate_steps = [('C', latest['C']), ('D', latest['D']), ('C', common), ('D', common)]
candidate_names = []
for arm, step in candidate_steps:
    key = f'{arm}_{step//1000}k'
    if key in curves:
        continue
    path = ROOT/arm/'joint'/f'validation_{step:07d}.json'
    pinned = OUT/'sources'/f'{arm}_{path.name}'
    shutil.copyfile(path, pinned)
    value = read(pinned)
    assert value == runs[arm]['validation'][step]
    assert value['_validation_identity'] == dict(dataset_fingerprint=dashboard.FINGERPRINT,
        images=24, crop=0, actual_rate=True)
    assert all(value[str(q)]['packet_forward_max_error'] == 0 for q in dashboard.QUALITIES)
    curves[key] = [dict(point, num_images=24) for point in dashboard.rd_curve(value)]
    sources[key] = dict(source=str(path), pinned=str(pinned), sha256=digest(pinned),
        step=step, weights='deployment_ema', maximum_packet_forward_error=0.)
    candidate_names.append(key)
for name, curve in curves.items():
    assert len(curve) == 6
    assert all(row['num_images'] == 24 for row in curve)
    assert np.isfinite([[row[k] for k in ('avg_bpp', *METRICS)] for row in curve]).all()
    assert np.all(np.diff([row['avg_bpp'] for row in curve]) > 0)

baseline_comparisons = {baseline: {name: compare(curves[baseline], curves[name])
    for name in candidate_names} for baseline in ('baseline_30k', 'historical_200k')}
paired = compare(curves[f'C_{common//1000}k'], curves[f'D_{common//1000}k'])
matched_rates = {str(rate): {name:interpolate(curve, rate) for name,curve in curves.items()}
                 for rate in (.032, .04, .06, .08, .10, .12, .15)}
bd_history = {baseline:{arm:[] for arm in ('C','D')} for baseline in ('baseline_30k','historical_200k')}
for baseline in bd_history:
    for arm in ('C','D'):
        for step, report in runs[arm]['validation'].items():
            bd_history[baseline][arm].append(dict(step=step,
                bd_rate=compare(curves[baseline], dashboard.rd_curve(report))))

# A dense numerical integral independently checks the reported log-rate integral.
numerical_errors = []
for baseline, candidates in baseline_comparisons.items():
    for name, comparisons in candidates.items():
        for metric, result in comparisons.items():
            if result['status'] != 'computed':
                continue
            xs = np.linspace(*result['increasing_quality_interval'], 20001)
            functions = []
            for model in (baseline, name):
                values = np.array([point[metric] for point in curves[model]])
                if metric in ('lpips','dists'):
                    values = -values
                functions.append(PchipInterpolator(values, np.log([p['avg_bpp'] for p in curves[model]])))
            approx = np.expm1(trapezoid(functions[1](xs)-functions[0](xs), xs)/(xs[-1]-xs[0]))*100
            error = abs(float(approx)-result['percent'])
            assert error < 1e-4, (baseline, name, metric, error)
            numerical_errors.append(error)

report = dict(updated_at=NOW, status=status, latest=latest, common_step=common,
    curves=curves, baseline_comparisons=baseline_comparisons, D_vs_C_same_step=paired,
    matched_rates=matched_rates, bd_rate_history=bd_history,
    candidate_sources=sources,
    baseline_sources=dict(local_30k=str(OLD/'baseline/all_summaries.json'),
        local_30k_sha256=digest(OLD/'baseline/all_summaries.json'),
        historical=str(OLD/'historical_reference.json'),
        historical_sha256=digest(OLD/'historical_reference.json'),
        workbook=history_reference['reference_workbook'],
        workbook_sha256=history_reference['reference_workbook_sha256'],
        historical_weights=history_reference['reference_weights']),
    bd_rate_method='PCHIP integral of log actual total bpp vs quality over each pair common quality interval; reverse sign for LPIPS/DISTS; no extrapolation and no nonmonotonic point removal. Positive means candidate needs more bits at equal metric.',
    numerical_integral_check_max_percentage_point_error=max(numerical_errors),
    evaluation_protocol=dict(current='EMA Kodak24 full-image training validation; hard-forward FP32 metrics; actual serialized packet bpp including all headers and side information for all 24 images at six qualities. Two images per quality also decoded and compared with hard forward; all errors zero.',
        historical_200k='512 training, no training window, global decode, EMA200000, archived Kodak24 true-rANS full-image float-output metrics. Original weights not available here; not rerun.',
        baseline_30k='Prior local 512 baseline, EMA30000, global decode, full true-rANS Kodak24 inference; reused.',
        prior_full_decode_check=old46['audit']),
    limitations=['Latest C and D have different validated step counts; matched-step C/D comparison is reported separately.',
        'Training budgets differ: C/D have a shared 50k codec warmup plus their current joint phase, versus distinct baseline schedules; not an equal-compute architecture comparison.',
        'Frontend initialization and image-loss weights differ: C/D 2/1/0.5 vs baseline 4/1/1.',
        'Current metrics come from validation, not a newly rerun full real-packet decoder sweep; the 46k full sweep and 66k six sample decodes agreed with hard forward.',
        'Kodak is used for checkpoint selection; no new independent dataset or multi-seed significance claim.',
        'Pairwise BD-rate quality integration intervals can differ; see overlap fractions.'])
(OUT/'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
with (OUT/'rd_points.csv').open('w') as fd:
    fields=('model','quality_index','avg_bpp', *METRICS)
    writer=csv.DictWriter(fd, fieldnames=fields)
    writer.writeheader()
    for name,curve in curves.items():
        for index,row in enumerate(curve):
            writer.writerow(dict(model=name,quality_index=index,**{key:row[key] for key in fields[2:]}))
with (OUT/'bd_rate.csv').open('w') as fd:
    writer=csv.writer(fd)
    writer.writerow(('candidate','reference', *METRICS))
    for reference, candidates in baseline_comparisons.items():
        for name, values in candidates.items():
            writer.writerow((name,reference,*[values[metric].get('percent') for metric in METRICS]))

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
    'axes.spines.top':False, 'axes.spines.right':False})
colors={'baseline_30k':'#9ca3af','historical_200k':'#111827',
        **{name:('#2563eb' if name.startswith('C_') else '#ea580c') for name in candidate_names}}
latest_names=[f'{arm}_{latest[arm]//1000}k' for arm in ('C','D')]
labels={'baseline_30k':'512 baseline / 30k (prior inference)',
        'historical_200k':'512 baseline / 200k (archived)',
        **{name:name.replace('_',' / ')+' (validation)' for name in candidate_names}}


def plot_rd(names, suffix, title):
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    for ax,metric in zip(axes,METRICS):
        for name in ('baseline_30k','historical_200k',*names):
            curve=curves[name]
            rates=np.array([point['avg_bpp'] for point in curve])
            dense=np.geomspace(rates[0],rates[-1],200)
            values=PchipInterpolator(np.log(rates),[row[metric] for row in curve])(np.log(dense))
            is_baseline=name in ('baseline_30k','historical_200k')
            ax.plot(dense,values,color=colors[name],linestyle='--' if is_baseline else '-',
                linewidth=1.8,label=labels[name])
            ax.scatter(rates,[row[metric] for row in curve],s=22,color=colors[name],
                marker='s' if is_baseline else ('o' if name.startswith('C') else '^'))
        ax.set_xlabel('Actual total bitrate (bpp)')
        ax.set_ylabel({'psnr':'PSNR (dB), higher is better','lpips':'LPIPS, lower is better','dists':'DISTS, lower is better'}[metric])
        ax.set_xlim(.023,.163)
        ax.grid(alpha=.2)
    fig.suptitle(title,x=.065,ha='left',fontsize=14)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,
        bbox_to_anchor=(.5,.03),frameon=False,fontsize=9)
    fig.text(.065,.01,'Kodak24 original images, EMA, float-output metrics. Current: validation + actual packet bpp; baselines: prior true-rANS inference. No extrapolation.',fontsize=8,color='#4b5563')
    fig.tight_layout(rect=(.01,.17,1,.92))
    fig.savefig(OUT/f'rd_curves{suffix}.png',dpi=180,facecolor='white')
    fig.savefig(OUT/f'rd_curves{suffix}.pdf',facecolor='white')
    plt.close(fig)


plot_rd(latest_names,'','Current C / D versus previous 512 baselines')
plot_rd([f'{arm}_{common//1000}k' for arm in ('C','D')], '_same_step',
    f'C / D matched at {common//1000}k joint updates versus previous 512 baselines')

# Show current gap directly; pairwise quality intervals are recorded in JSON.
fig,axes=plt.subplots(1,3,figsize=(12.5,4.6))
for ax,metric in zip(axes,METRICS):
    values=[baseline_comparisons['historical_200k'][name][metric].get('percent',np.nan) for name in candidate_names]
    bars=ax.bar(np.arange(len(candidate_names)),values,color=[colors[name] for name in candidate_names],width=.55)
    ax.axhline(0,color='#111827',linewidth=1)
    ax.bar_label(bars,labels=[f'{value:+.2f}%' for value in values],padding=4,fontsize=10)
    ax.set_xticks(np.arange(len(candidate_names)),[name.replace('_',' / ') for name in candidate_names])
    ax.set_ylabel('BD-rate vs 512 baseline 200k (%)')
    ax.set_title(metric.upper(),loc='left')
    ax.set_ylim(min(-3.,min(values)-3), max(values)*1.22)
    ax.grid(axis='y',alpha=.18)
fig.suptitle('Gap to 512 baseline / 200k: positive = more bits needed',x=.065,ha='left',fontsize=14)
fig.text(.065,.02,'PCHIP over common quality interval, no extrapolation; step labels refer to the C/D joint phase. Baseline is an archived result.',fontsize=8.5,color='#4b5563')
fig.tight_layout(rect=(.01,.06,1,.9))
fig.savefig(OUT/'bd_rate_vs_baseline200k.png',dpi=180,facecolor='white')
fig.savefig(OUT/'bd_rate_vs_baseline200k.pdf',facecolor='white')
plt.close(fig)

# Follow progress with a fixed baseline instead of comparing raw training loss.
fig,axes=plt.subplots(2,3,figsize=(14,8))
for index, baseline in enumerate(('historical_200k','baseline_30k')):
    for ax,metric in zip(axes[index],METRICS):
        for arm in ('C','D'):
            points=[row for row in bd_history[baseline][arm] if row['bd_rate'][metric]['status']=='computed']
            x=[row['step']/1000 for row in points]
            y=[row['bd_rate'][metric]['percent'] for row in points]
            ax.plot(x,y,'.-',color='#2563eb' if arm=='C' else '#ea580c',label=arm,linewidth=1.5)
        ax.axhline(0,color='#374151',linewidth=1)
        ax.set_title(f'{metric.upper()} vs baseline {"200k" if index==0 else "30k"}',loc='left')
        ax.set_xlabel('Joint updates (thousands)')
        ax.set_ylabel('BD-rate (%)')
        ax.grid(alpha=.18)
        ax.legend(frameon=False)
fig.suptitle('Training progress relative to fixed 512 baselines | lower is better',x=.065,ha='left',fontsize=15)
fig.text(.065,.018,'Raw validation checkpoints, no smoothing. Positive = candidate needs more bitrate; negative = less. Pairwise common quality intervals may change.',fontsize=9,color='#4b5563')
fig.tight_layout(rect=(.01,.045,1,.94),h_pad=2)
fig.savefig(OUT/'bd_rate_progress.png',dpi=170,facecolor='white')
fig.savefig(OUT/'bd_rate_progress.pdf',facecolor='white')
plt.close(fig)

lines=['# C / D 当前水平与 512 baseline 对比', '', f'快照时间：{NOW}', '',
    '以下依据已保存日志与验证文件；未核验当前 GPU/进程运行状态。checkpoint 步数均指 C/D 的 joint 阶段，之前还各有共同的 50k codec warmup。', '',
    '| 模型 | 当前训练 step | 最新验证 step | 当前阶段预算 | 最新验证损失 | 最佳验证 step |',
    '|---|---:|---:|---:|---:|---:|']
for arm,value in status.items():
    lines.append(f'| {arm} | {value["train_step"]} | {latest[arm]} | 200000 | {value["validation_rd_mean"]:.6f} | {value["best"]["step"]} |')
lines += ['', '主表采用各模型最新完整验证。正 BD-rate＝同指标需更多码率（较差）；负值＝省码率（较好）。共同质量区间 PCHIP 积分，不外推。', '',
    '| 模型 | 参照 | PSNR BD-rate | LPIPS BD-rate | DISTS BD-rate |', '|---|---|---:|---:|---:|']
for baseline, label in (('historical_200k','512 / 200k（历史归档）'),('baseline_30k','512 / 30k（之前本机推理）')):
    for name in latest_names:
        values=baseline_comparisons[baseline][name]
        lines.append('| '+' | '.join([name.replace('_',' / '),label,*[bd_text(values[m]) for m in METRICS]])+' |')
lines += ['', f'C/D 同步数比较（双方 {common//1000}k）：', '',
    '| 比较 | PSNR BD-rate | LPIPS BD-rate | DISTS BD-rate |', '|---|---:|---:|---:|',
    '| D 相对 C | '+' | '.join(bd_text(paired[m]) for m in METRICS)+' |', '',
    f'最新验证步数 C={latest["C"]}、D={latest["D"]}；公平 C/D 比较使用双方同为 {common//1000}k 的附图和上述同步数行。当前 D 相对老 30k baseline 有优势，对历史 200k baseline 尚有差距。', '',
    f'C 的最佳综合验证 step={status["C"]["best"]["step"]}，D 的最佳综合验证 step={status["D"]["best"]["step"]}。近期处于缓慢改善和波动阶段；综合 loss 最低不表示每个画质指标都同时最好，尚不能据此断言已完全收敛或必定赶上 200k baseline。', '',
    '本轮评测来源：C/D 是最新 EMA 的 Kodak24 原图验证，画质取 FP32 硬前向、bpp 来自全部完整真实码流字节，每档另抽 2 张解码核对；本次所用验证的核对误差均为 0。没有把本轮称为全量真实解码复测。30k baseline 沿用之前本机全量真实 rANS 推理；200k baseline 是之前 512 非窗口训练、全局解码的历史记录，权重在本机不可用，未重跑。46k 曾完成全量 C/D 真实解码并与验证一致，66k 六张样例真实解码也一致。', '',
    '训练总预算、前端初始化和损失不同（C/D 图像损失 2/1/0.5，baseline 为 4/1/1），这里衡量当前结果差距。Kodak 用于模型选择，不能当作独立测试或等预算的结构优劣结论。', '',
    f'![最新 RD 曲线]({OUT/"rd_curves.png"})', '',
    f'![同为{common//1000}k RD曲线]({OUT/"rd_curves_same_step.png"})', '',
    f'![与200k baseline的差距]({OUT/"bd_rate_vs_baseline200k.png"})', '',
    f'![相对 baseline 的训练趋势]({OUT/"bd_rate_progress.png"})', '',
    '完整六档数据、各指标积分区间与覆盖比例见 comparison.json；可编辑数值见 rd_points.csv 和 bd_rate.csv。']
(OUT/'report.zh.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:30]))
print('Numerical integration maximum error (percentage points):',max(numerical_errors))
