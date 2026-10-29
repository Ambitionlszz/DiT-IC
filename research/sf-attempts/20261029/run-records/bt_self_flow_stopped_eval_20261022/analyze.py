"""Audit stopped C/D convergence and real serialized-packet RD results."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import struct

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get('BT_EVAL_DATA', '/tmp/bt-self-flow-stopped-eval-20261022'))
OLD = Path('/data2/ls/checkpoints/bt_self_flow_v1')
ARCHIVE = Path('/data2/ls/checkpoints/bt_self_flow_comparison_20261019')
METRICS = ('psnr', 'ms_ssim', 'lpips', 'dists')
COLORS = {'C': '#2563eb', 'D': '#e36a17', 'Baseline 512 / 200K': '#222222'}


def read(path):
    return json.loads(path.read_text())


def interp(rows, metric, bpp=.06, rate='avg_bpp'):
    rows = sorted(rows, key=lambda r: r[rate])
    x = np.log([r[rate] for r in rows])
    if not x[0] <= np.log(bpp) <= x[-1]:
        raise ValueError('Extrapolation is forbidden')
    return float(PchipInterpolator(x, [r[metric] for r in rows])(np.log(bpp)))


def convergence():
    history = {}
    flat = []
    for name in ('C', 'D'):
        history[name] = []
        for path in sorted((OLD/name/'joint').glob('validation_*.json')):
            d = read(path)
            assert d['_validation_identity']['images'] == 24
            assert d['_validation_identity']['crop'] == 0
            points = [v for k, v in sorted(d.items()) if not k.startswith('_')]
            assert len(points) == 6
            s = int(path.stem.split('_')[1])
            row = {'model': name, 'step': s}
            row.update({k: float(np.mean([p[k] for p in points]))
                        for k in ('loss', 'psnr', 'lpips', 'dists', 'base', 'latent', 'rate_actual')})
            for bpp in (.04, .06, .08):
                for metric in ('psnr', 'lpips', 'dists'):
                    row[f'{metric}_at_{bpp}'] = interp(points, metric, bpp, 'rate_actual')
            history[name].append(row)
            flat.append(row)
    with (ROOT/'validation_trend.csv').open('w') as fd:
        writer = csv.DictWriter(fd, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    (ROOT/'convergence.json').write_text(json.dumps(history, indent=2)+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, key, label in zip(axes.flat,
            ('loss', 'psnr_at_0.06', 'lpips_at_0.06', 'dists_at_0.06'),
            ('Mean validation RD loss (six fixed qualities)', 'PSNR at 0.06 bpp (dB)',
             'LPIPS at 0.06 bpp (lower is better)', 'DISTS at 0.06 bpp (lower is better)')):
        for name, rows in history.items():
            ax.plot([r['step']/1000 for r in rows], [r[key] for r in rows],
                    color=COLORS[name], lw=1.7, label=name)
        ax.axvline(80, color='grey', ls=':', lw=1)
        ax.set(xlabel='Joint training updates (K)', ylabel=label)
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('Stopped legacy C/D: Kodak24 full-image EMA validation\n'
                 'Fixed-rate values: PCHIP in log actual bpp; validation is separate from packet inference')
    fig.savefig(ROOT/'convergence.png', dpi=170)
    plt.close(fig)
    for name, rows in history.items():
        start = next(r for r in rows if r['step'] == 80000)
        end = rows[-1]
        print(name, '80K to latest:', {k: end[k]-start[k] for k in
              ('loss', 'psnr_at_0.06', 'lpips_at_0.06', 'dists_at_0.06')})
    return history


def allocation_stats(name):
    histograms = {}
    per_image = {}
    header = struct.Struct('>4sBBB6I2H')
    for path in sorted((DATA/name).glob('q*/*.btic')):
        data = path.read_bytes()
        fields = header.unpack_from(data)
        lengths = struct.unpack_from('>5I', data, header.size)
        count = fields[6]*fields[7]
        start = header.size+20+lengths[0]+lengths[1]
        allocation = data[start:start+lengths[2]]
        assert len(allocation) == (3*count+7)//8
        bits = int.from_bytes(allocation, 'big')
        padding = len(allocation)*8-3*count
        codes = [(bits >> (padding+3*(count-1-i))) & 7 for i in range(count)]
        histograms.setdefault(path.parent.name, np.zeros(8, dtype=np.int64))
        histograms[path.parent.name] += np.bincount(codes, minlength=8)
        per_image.setdefault(path.stem, []).append(codes)
    assert len(histograms) == 6 and all(sum(h) == 2304 for h in histograms.values())
    return {'histograms': {q: h.tolist() for q, h in histograms.items()},
            'fraction_code_3_or_4': {q: float((h[3]+h[4])/sum(h)) for q, h in histograms.items()},
            'images_with_identical_allocation_across_six_q': sum(all(c == v[0] for c in v) for v in per_image.values()),
            'images': len(per_image)}


def reconstructions():
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    canvas = Image.new('RGB', (1140, 3*298+64), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), 'Real decoded PNG crops, q=0.4; individual bpp are shown (not exactly matched rate).', fill='black', font=font)
    draw.text((14, 33), 'The quality metrics above use float reconstruction before PNG rounding.', fill='black', font=font)
    for i, (stem, box) in enumerate((('kodim01', (160, 140, 544, 396)),
                                   ('kodim13', (200, 160, 584, 416)),
                                   ('kodim23', (160, 120, 544, 376)))):
        original = Image.open(Path('/data2/ls/data/Kodak')/(stem+'.png')).convert('RGB')
        for j, name in enumerate(('Original', 'C', 'D')):
            y = 64+i*298
            if name == 'Original':
                image = original
                label = stem+' / original'
            else:
                rows = [json.loads(s) for s in (DATA/name/'images.jsonl').read_text().splitlines()]
                row = next(r for r in rows if Path(r['image']).stem == stem and r['lambda_code']==26214)
                p = next((DATA/name/'q26214').glob('*_'+stem+'.png'))
                image = Image.open(p).convert('RGB')
                label = f"{name} / {row['bpp']:.5f} bpp / {row['psnr']:.2f} dB"
            draw.text((j*380+8, y), label, fill='black', font=font)
            canvas.paste(image.crop(box).resize((376, 251)), (j*380, y+28))
    canvas.save(ROOT/'reconstruction_crops.png')


def inference(history, manifests=None):
    manifests = manifests or {name: read(DATA/name/'summary.json') for name in ('C', 'D')}
    assert manifests['C']['dataset_fingerprint'] == manifests['D']['dataset_fingerprint']
    assert manifests['C']['metric_protocol'] == manifests['D']['metric_protocol']
    curves = {}
    for name, manifest in manifests.items():
        assert manifest['images'] == 24 and manifest['weights'] == 'ema' and manifest['nfe'] == 1
        rows = [json.loads(s) for s in (DATA/name/'images.jsonl').read_text().splitlines()]
        assert len(rows) == 144
        for row in rows:
            assert row['nfe'] == 1
            assert sum(row['bytes_by_component'].values()) == row['bytes']
            assert np.isfinite([row[k] for k in ('bpp', *METRICS)]).all()
            assert abs(row['bpp']-8*row['bytes']/(row['height']*row['width'])) < 1e-12
            packet = next((DATA/name/f"q{row['lambda_code']:05d}").glob(f"*_{Path(row['image']).stem}.btic"))
            assert packet.stat().st_size == row['bytes']
        curves[name] = [dict(v, avg_bpp=v['bpp'], quality_index=int(k)/65535)
                        for k, v in manifest['summary'].items()]
    hist = read(ARCHIVE/'historical_reference.json')
    curves['Baseline 512 / 200K'] = hist['reference_curves']['A：非窗口训练 / 全局']
    spec = importlib.util.spec_from_file_location('bd', '/data2/ls/code/deployment-bootstrap/compare_baseline512_history.py')
    bd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bd)
    comparisons = {}
    for candidate, reference in (('C', 'Baseline 512 / 200K'), ('D', 'Baseline 512 / 200K'), ('D', 'C')):
        comparisons[f'{candidate}_vs_{reference}'] = {m: bd.bd_rate(curves[reference], curves[candidate], m) for m in METRICS}
    matched = {str(rate): {name: {m: interp(rows, m, rate) for m in METRICS}
                          for name, rows in curves.items()}
               for rate in (.04, .06, .08)}
    result = {'steps': {k: v['step'] for k, v in manifests.items()}, 'curves': curves,
              'bd_rate': comparisons, 'matched_rate_interpolation': matched,
              'allocation_stats': {name: allocation_stats(name) for name in ('C', 'D')},
              'method': 'PCHIP log actual total packet bpp over common quality interval; no extrapolation, no point deletion; raw MS-SSIM',
              'limits': ['C/D have slightly different training steps; this is their stopped deployment comparison, not an equal-step D ablation.',
                         'The 200K baseline curve is archived, not rerun today. Different training objectives and budgets prevent architecture-only attribution.',
                         'Timing runs concurrently with new warmup and is unsuitable for speed comparisons.'],
              'baseline_source': str(ARCHIVE/'historical_reference.json'),
              'baseline_checkpoint_identity': hist['reference_weights'][1]}
    (ROOT/'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    with (ROOT/'rd_points.csv').open('w') as fd:
        writer = csv.DictWriter(fd, fieldnames=['model', 'quality_index', 'avg_bpp', *METRICS])
        writer.writeheader()
        for name, rows in curves.items():
            writer.writerows(dict(model=name, **{k: r[k] for k in writer.fieldnames[1:]}) for r in rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, m, label in zip(axes.flat, METRICS, ('PSNR (dB)', 'MS-SSIM', 'LPIPS (lower is better)', 'DISTS (lower is better)')):
        for name, rows in curves.items():
            rows = sorted(rows, key=lambda r: r['avg_bpp'])
            step = f" / {manifests[name]['step']/1000:.2f}K" if name in manifests else ' (archived)'
            ax.plot([r['avg_bpp'] for r in rows], [r[m] for r in rows], 'o-', color=COLORS[name],
                    label=name+step, lw=1.7, ms=4)
        ax.set(xlabel='Actual total packet bpp', ylabel=label)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Kodak24 / full images / real rANS packets / EMA / FP32 / global 1-NFE\n'
                 'PSNR, MS-SSIM, DISTS: pyiqa; LPIPS: torchmetrics alex; all header bytes billed')
    fig.savefig(ROOT/'rd_curves.png', dpi=170)
    plt.close(fig)
    reconstructions()
    lines = ['# 旧 C/D 停止后的完整图真实码流评测', '',
             f"C：{manifests['C']['step']:,} 步；D：{manifests['D']['step']:,} 步。均加载停止时 paused.pt 的 EMA。", '',
             'Kodak24 原尺寸完整图、六档、FP32、全局一次 SANA 解码。真正序列化后再从包解码，bpp 计入所有包头和分配图。',
             '指标与已有 512 评测脚本使用同一实现：pyiqa PSNR/MS-SSIM/DISTS，torchmetrics LPIPS alex。历史 200K baseline 使用已归档的全局解码结果，本次没有重跑 baseline。', '',
             '## 收敛趋势', '']
    for name, rows in history.items():
        a = next(r for r in rows if r['step'] == 80000)
        b = rows[-1]
        lines.append(f"- {name}：80K→{b['step']/1000:g}K，六档平均验证 loss {a['loss']:.6f}→{b['loss']:.6f}（{(b['loss']/a['loss']-1)*100:+.2f}%）；0.06 bpp 插值 PSNR {a['psnr_at_0.06']:.4f}→{b['psnr_at_0.06']:.4f} dB，LPIPS {a['lpips_at_0.06']:.6f}→{b['lpips_at_0.06']:.6f}，DISTS {a['dists_at_0.06']:.6f}→{b['dists_at_0.06']:.6f}。")
    lines += ['', '这些曲线表现为趋于平台，感知指标仍缓慢改善；不能据此保证达到数学收敛，也没有证据表明继续到 200K 会自然追平 baseline。', '',
              '## BD-rate', '', '正值表示同质量需要更多码率。PCHIP、共同质量区间、不外推；MS-SSIM 按原值积分。', '',
              '| 候选 / 参考 | PSNR | MS-SSIM | LPIPS | DISTS |', '|---|---:|---:|---:|---:|']
    for pair, values in comparisons.items():
        numbers = [f"{values[m]['percent']:+.2f}%" if values[m]['status']=='computed' else 'N/A' for m in METRICS]
        lines.append('| '+' | '.join([pair, *numbers])+' |')
    lines += ['', '## 同码率插值（用于观察差距，不是这些精确 bpp 的新增实测点）', '',
              '| bpp | 模型 | PSNR | MS-SSIM | LPIPS | DISTS |', '|---:|---|---:|---:|---:|---:|']
    for rate, groups in matched.items():
        for name, metrics in groups.items():
            lines.append('| '+' | '.join([rate, name, *[f'{metrics[m]:.5f}' for m in METRICS]])+' |')
    lines += ['', '## 码流中的 Router 分配', '']
    for name, stats in result['allocation_stats'].items():
        h = stats['histograms']['q00000']
        major = int(np.argmax(h))
        lines.append(f"- {name}：主要编码值 {major} 占 {h[major]/sum(h)*100:.2f}%；完整计数 {h}。24/24 张图的分配图在六个 q 下完全相同，与旧 Router 没有 q 条件的实现一致。")
    lines += ['', '编码值本身经过逐图去均值后才转换成局部步长；主要编码值不同不代表全局步长发生偏移。分配是否有价值不能仅靠使用档位的多少判断。', '', '## 解释边界', '',
              '- C/D 停止步数相差 1,208 步，结果代表这两个停止时模型，不能当严格同训练步数的 D 消融。',
              '- 旧 C/D 使用 2×MSE+LPIPS+0.5×DISTS，并有 0.1 base/0.1 repaired 辅助项；历史 512 使用 4×MSE+LPIPS+DISTS。训练目标与训练预算不同。',
              '- GPU 推理与新 warmup 并行，时间数据不能用于速度排名。新训练没有被停止。',
              '- 完整逐图码流与代表性 PNG 归档在 real_packet_results.tar.gz；全部 PNG 的完整归档保留在本机 /tmp/bt-stopped-real-packet-results-20261022.tar.gz。逐图指标、曲线及 BD-rate 积分区间保存在本目录。checkpoint SHA256 见 C/summary.json 与 D/summary.json。', '']
    (ROOT/'report.zh.md').write_text('\n'.join(lines))
    print(json.dumps({'steps':result['steps'], 'bd_rate':comparisons, 'matched':matched}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    h = read(ROOT/'convergence.json') if (ROOT/'convergence.json').exists() else convergence()
    if all((ROOT/name/'summary.json').exists() for name in ('C','D')):
        inference(h)
    else:
        print('Full packet inference is still pending; convergence artifacts saved.')
