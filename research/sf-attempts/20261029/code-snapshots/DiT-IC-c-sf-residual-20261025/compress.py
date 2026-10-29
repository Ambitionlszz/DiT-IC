import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torchvision import transforms
from torchmetrics.image import (
    FrechetInceptionDistance,
    KernelInceptionDistance,
    LearnedPerceptualImagePatchSimilarity,
)

import os
import math
import json
import glob
import yaml
import pyiqa
import sys
import datetime
import importlib
import numpy as np
from PIL import Image
from accelerate.utils import set_seed
import time
import gc
from pathlib import Path

from codec_packet import (
    PACKET_MAGIC,
    CodecPacket,
    read_codec_packet,
    write_codec_packet,
)
from eval.compress_utils import *
from eval.testing_utils import parse_args_testing
from eval._update_patch_fid import update_patch_fid
from config_compat import load_pure_sana_config, prepare_inference_config

class TeeLogger:
    """
    Duplicates stdout to both the original stdout and a log file.
    Sets up automatic cleanup on exit.
    """

    def __init__(self, log_path):
        self.original_stdout = sys.stdout
        self.log_file = open(log_path, 'w', encoding='utf-8')

    def write(self, message):
        self.original_stdout.write(message)
        self.log_file.write(message)

    def flush(self):
        self.original_stdout.flush()
        self.log_file.flush()

    def close(self):
        sys.stdout = self.original_stdout
        self.log_file.close()


def setup_logging(results_dir, run_name):
    """Setup a TeeLogger to capture all stdout to a log file."""
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "inference.log")
    tee = TeeLogger(log_path)
    sys.stdout = tee
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Inference Run: {run_name}")
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Log file: {log_path}")
    print()
    return tee


def codec_models(model, device, **kwargs):
    model_map = {
        'Codec': 'models.DiT_IC.Codec',

    }

    if model not in model_map:
        raise ValueError(f"Unknown model name: {model}")

    module_path, class_name = model_map[model].rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)(device, model, **kwargs)


def load_config(config_path):
    return load_pure_sana_config(config_path)


def print_model_parameters(net, model_name="DiT-IC"):
    """
    打印模型各模块的参数量统计信息
    
    Args:
        net: 模型实例
        model_name: 模型名称（用于显示）
    """
    print(f"\n{'='*80}")
    print(f"Model Parameters Statistics - {model_name}")
    print(f"{'='*80}\n")
    
    # 总参数量
    total_params = sum(p.numel() for p in net.parameters())
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total_params_m = total_params / 1e6
    total_trainable_params_m = total_trainable_params / 1e6
    
    print(f"Total Parameters:     {total_params_m:.2f}M ({total_params:,})")
    print(f"Trainable Parameters: {total_trainable_params_m:.2f}M ({total_trainable_params:,})")
    print(f"{'-'*80}\n")
    
    # 按主要模块分组统计
    module_stats = {}
    
    # 遍历顶层模块
    for name, module in net.named_children():
        params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        module_stats[name] = {
            'params': params,
            'trainable_params': trainable_params,
            'module': module
        }
    
    # 按参数量排序
    sorted_modules = sorted(module_stats.items(), key=lambda x: x[1]['params'], reverse=True)
    
    print(f"Module-wise Breakdown:")
    print(f"{'Module Name':<40} {'Params (M)':>12} {'Trainable (M)':>15} {'Percentage':>12}")
    print(f"{'-'*80}")
    
    for name, stats in sorted_modules:
        params_m = stats['params'] / 1e6
        trainable_m = stats['trainable_params'] / 1e6
        percentage = (stats['params'] / total_params * 100) if total_params > 0 else 0
        
        print(f"{name:<40} {params_m:>12.2f} {trainable_m:>15.2f} {percentage:>11.2f}%")
    
    print(f"{'-'*80}")
    print(f"{'Total':<40} {total_params_m:>12.2f} {total_trainable_params_m:>15.2f} {'100.00%':>12}")
    print(f"{'='*80}\n")
    
    # 如果有codec属性，进一步详细统计codec内部模块
    if hasattr(net, 'codec') and net.codec is not None:
        print(f"Codec Internal Modules Details:")
        print(f"{'='*80}\n")
        
        codec = net.codec
        codec_module_stats = {}
        
        for name, module in codec.named_children():
            params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            codec_module_stats[name] = {
                'params': params,
                'trainable_params': trainable_params
            }
        
        # 按参数量排序
        sorted_codec_modules = sorted(codec_module_stats.items(), key=lambda x: x[1]['params'], reverse=True)
        
        print(f"{'Module Name':<40} {'Params (M)':>12} {'Trainable (M)':>15} {'Percentage':>12}")
        print(f"{'-'*80}")
        
        for name, stats in sorted_codec_modules:
            params_m = stats['params'] / 1e6
            trainable_m = stats['trainable_params'] / 1e6
            percentage = (stats['params'] / total_params * 100) if total_params > 0 else 0
            
            print(f"{name:<40} {params_m:>12.2f} {trainable_m:>15.2f} {percentage:>11.2f}%")
        
        print(f"{'='*80}\n")
        
        # 如果codec有latent_codec，进一步统计
        if hasattr(codec, 'latent_codec') and codec.latent_codec is not None:
            print(f"LatentCodec Internal Modules Details:")
            print(f"{'='*80}\n")
            
            latent_codec = codec.latent_codec
            lc_module_stats = {}
            
            for name, module in latent_codec.named_children():
                params = sum(p.numel() for p in module.parameters())
                trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                lc_module_stats[name] = {
                    'params': params,
                    'trainable_params': trainable_params
                }
            
            # 按参数量排序
            sorted_lc_modules = sorted(lc_module_stats.items(), key=lambda x: x[1]['params'], reverse=True)
            
            print(f"{'Module Name':<40} {'Params (M)':>12} {'Trainable (M)':>15} {'Percentage':>12}")
            print(f"{'-'*80}")
            
            for name, stats in sorted_lc_modules:
                params_m = stats['params'] / 1e6
                trainable_m = stats['trainable_params'] / 1e6
                percentage = (stats['params'] / total_params * 100) if total_params > 0 else 0
                
                print(f"{name:<40} {params_m:>12.2f} {trainable_m:>15.2f} {percentage:>11.2f}%")
            
            print(f"{'='*80}\n")
    
    # 如果有transformer/dit模块，单独统计
    if hasattr(net, 'transformer') and net.transformer is not None:
        print(f"Transformer/DiT Module Details:")
        print(f"{'='*80}\n")
        
        transformer = net.transformer
        trans_params = sum(p.numel() for p in transformer.parameters())
        trans_trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        trans_params_m = trans_params / 1e6
        trans_trainable_params_m = trans_trainable_params / 1e6
        percentage = (trans_params / total_params * 100) if total_params > 0 else 0
        
        print(f"Transformer Total Params:     {trans_params_m:.2f}M ({trans_params:,})")
        print(f"Transformer Trainable Params: {trans_trainable_params_m:.2f}M ({trans_trainable_params:,})")
        print(f"Percentage of Total:          {percentage:.2f}%")
        print(f"{'='*80}\n")


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image_tensor


def compress_one_image(
    net, bin_path, ori_h, ori_w, img_name, x, quality_index=None
):
    with torch.no_grad():
        packet_or_dict = net.compress(x, quality_index)
    if not os.path.exists(bin_path): os.makedirs(bin_path)
    output_path = os.path.join(bin_path, img_name)
    with Path(output_path).open("wb") as f:
        if isinstance(packet_or_dict, CodecPacket):
            write_codec_packet(f, packet_or_dict)
        else:
            write_body(
                f,
                packet_or_dict["shape"],
                packet_or_dict["strings"],
            )
    size = filesize(output_path)
    bpp = float(size) * 8 / (ori_h * ori_w)
    return bpp


def decompress_one_image(net, bin_path, ori_h, ori_w, img_name):
    output = os.path.join(bin_path, img_name)
    with Path(output).open("rb") as f:
        magic = f.read(len(PACKET_MAGIC))
        f.seek(0)
        if magic == PACKET_MAGIC:
            packet = read_codec_packet(f)
        else:
            packet = None
            strings, shape = read_body(f)
    with torch.no_grad():
        if packet is not None:
            out_img = net.decompress(packet)
        else:
            out_img = net.decompress(strings, shape)
    out_img = out_img[:, :, 0 : ori_h, 0 : ori_w]
    out_img = (out_img * 0.5 + 0.5).float().detach()
    return out_img


def measure_text_encoder_memory(text_encoder_path, device='cuda', continue_inference=False):
    """
    测量文本编码器的显存占用（加载但不使用）
    text_encoder_path: 应该指向包含 model_index.json 的模型根目录（如 ./SANA）
    continue_inference: 如果为 True，加载后不删除，返回 text_encoder 供后续使用
    """
    print(f"\n{'='*60}")
    print(f"Measuring Text Encoder Memory Usage")
    print(f"{'='*60}")
    
    # 检查路径
    if not os.path.exists(text_encoder_path):
        print(f"❌ Error: Path does not exist: {text_encoder_path}")
        return None, None
    
    # 检查是否错误地指向了 text_encoder 子文件夹
    if os.path.exists(os.path.join(text_encoder_path, 'config.json')) and not os.path.exists(os.path.join(text_encoder_path, 'model_index.json')):
        print(f"⚠️  Warning: The path seems to point to a subcomponent.")
        print(f"   Please provide the path to the model root directory containing 'model_index.json'.")
        # 尝试自动修正
        parent_path = os.path.dirname(text_encoder_path)
        if os.path.exists(os.path.join(parent_path, 'model_index.json')):
            print(f"✓ Auto-correcting to: {parent_path}\n")
            text_encoder_path = parent_path
        else:
            print(f"❌ Cannot auto-correct. Please specify correct model root path.\n")
            return None, None
    
    if not os.path.exists(os.path.join(text_encoder_path, 'model_index.json')):
        print(f"❌ Error: model_index.json not found in {text_encoder_path}")
        return None, None
    
    print(f"Model path: {text_encoder_path}")
    print(f"Device: {device}\n")
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    gc.collect()
    
    mem_before = torch.cuda.memory_allocated(device) / (1024 ** 2)
    print(f"Memory before loading text encoder: {mem_before:.2f} MB")
    
    # Load text encoder
    try:
        from diffusers import SanaPipeline
        
        print(f"\nLoading text encoder from: {text_encoder_path}")
        print("This may take a while...\n")
        
        # Record time
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Load only text encoder
        text_encoder = SanaPipeline.from_pretrained(
            text_encoder_path,
            torch_dtype=torch.float32,  # 使用 FP32 测量真实显存
        ).text_encoder
        
        # Move to GPU
        text_encoder = text_encoder.to(device)
        text_encoder.eval()
        
        torch.cuda.synchronize()
        load_time = time.time() - start_time
        
        # Measure memory
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated(device) / (1024 ** 2)
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        
        text_encoder_mem = mem_after - mem_before
        num_params = sum(p.numel() for p in text_encoder.parameters())
        num_params_m = num_params / 1e6
        
        print(f"{'='*60}")
        print(f"Text Encoder Memory Statistics:")
        print(f"{'='*60}")
        print(f"  Load time:          {load_time:.2f}s")
        print(f"  Memory before:      {mem_before:.2f} MB")
        print(f"  Memory after:       {mem_after:.2f} MB")
        print(f"  Text encoder used:  {text_encoder_mem:.2f} MB")
        print(f"  Peak memory:        {peak_mem:.2f} MB")
        print(f"  Parameters:         {num_params_m:.2f}M")
        print(f"  Model class:        {text_encoder.__class__.__name__}")
        
        # Count layers
        total_layers = sum(1 for _ in text_encoder.modules())
        linear_layers = sum(1 for m in text_encoder.modules() if isinstance(m, torch.nn.Linear))
        print(f"  Total modules:      {total_layers}")
        print(f"  Linear layers:      {linear_layers}")
        
        if continue_inference:
            # 如果不继续推理，才清理
            print(f"\n✓ Text encoder loaded and kept in memory for inference comparison.\n")
            print(f"{'='*60}\n")
            return text_encoder, {
                'memory_mb': text_encoder_mem,
                'peak_memory_mb': peak_mem,
                'params_m': num_params_m,
                'load_time': load_time
            }
        else:
            # Cleanup
            del text_encoder
            torch.cuda.empty_cache()
            gc.collect()
            
            print(f"\nMemory after cleanup: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")
            print(f"{'='*60}\n")
            
            return None, {
                'memory_mb': text_encoder_mem,
                'peak_memory_mb': peak_mem,
                'params_m': num_params_m,
                'load_time': load_time
            }
        
    except Exception as e:
        print(f"\n❌ Error loading text encoder: {e}")
        print(f"\n可能需要的操作:")
        print(f"  1. 确保已下载 text_encoder 文件夹到: {text_encoder_path}")
        print(f"  2. 检查 diffusers 版本是否支持 SanaPipeline")
        print(f"  3. 确认 text_encoder 目录包含正确的权重文件")
        import traceback
        traceback.print_exc()
        return None, None

def main(args):

    if args.seed is not None:
        set_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    print(f"Using GPU: {args.gpu} ({torch.cuda.get_device_name(device)})")

    # A legacy training YAML is also a valid inference YAML. CLI values only
    # override fields when they were explicitly provided.
    config = prepare_inference_config(
        load_config(args.config_path),
        codec_path=args.codec_path,
        use_ema=args.use_ema,
        use_merge=args.use_merge,
    )
    params = config["model"]["params"]
    args.codec_path = params["codec_path"]
    variable_rate = bool(
        (params.get("codec_param") or {}).get("variable_rate", False)
    )
    if variable_rate and args.quality_index is None:
        raise ValueError("--quality_index is required for a variable-rate codec")
    if not variable_rate and args.quality_index is not None:
        raise ValueError("--quality_index requires codec_param.variable_rate=true")

    # ===== Setup results directory and logging =====
    ckpt_name = Path(args.codec_path).stem if args.codec_path else "no_ckpt"
    config_name = Path(args.config_path).stem
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    quality_suffix = (
        f"_q{args.quality_index}" if args.quality_index is not None else ""
    )
    run_name = f"{config_name}_{ckpt_name}{quality_suffix}_{timestamp}"
    run_dir = os.path.join(args.results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_tee = setup_logging(run_dir, run_name)
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Results dir: {run_dir}")
    print()

    # Save the effective config for reproducibility
    try:
        effective_config_path = os.path.join(run_dir, "config_used.yaml")
        with open(effective_config_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False)
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Config saved to: {effective_config_path}")
    except Exception as e:
        print(f"Warning: Could not save config copy: {e}")
    # ===============================================

    # ===== 新增：测量文本编码器显存 =====
    text_encoder = None
    if args.measure_text_encoder:
        text_encoder_path = args.text_encoder_path if args.text_encoder_path else args.config_path.replace('configs/', '').replace('.yaml', '')
        text_encoder, stats = measure_text_encoder_memory(text_encoder_path, device, continue_inference=args.continue_inference)
        if not args.continue_inference:
            log_tee.close()
            return  # 如果只测量文本编码器，不继续推理
    # ====================================
    
    # define Non-Reference Model
    metric_dict = {}
    metric_dict["clipiqa"] = pyiqa.create_metric('clipiqa').to(device) # [0-1] input
    metric_dict["musiq"] = pyiqa.create_metric('musiq').to(device) # [0-1] input
    metric_dict["niqe"] = pyiqa.create_metric('niqe').to(device) # [0-1] input
    # metric_dict["maniqa"] = pyiqa.create_metric('maniqa').to(device) # [0-1] input
    
    # define Reference Model
    metric_paired_dict = {}
    metric_paired_dict["psnr"] = pyiqa.create_metric('psnr').to(device) # [0-1] input
    metric_paired_dict["dists"] = pyiqa.create_metric('dists').to(device) # [0-1] input
    metric_paired_dict["ms_ssim"] = pyiqa.create_metric('ms_ssim').to(device) # [0-1] input
    metric_paired_dict["lpips"] = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device) # [0-1] input
    
    # define Distribution Model
    fid_metric = FrechetInceptionDistance().to(device)
    kid_metric = KernelInceptionDistance().to(device)
    
    # load model
    params = config.get("model", {}).get("params", {})
    net = codec_models(config['model']['model_type'], 'cuda', **params)
    net.cuda().eval()
    net.codec.update(force=True)
    
    # ===== 新增：打印模型各模块参数量统计 =====
    print_model_parameters(net, model_name=config['model']['model_type'])
    # ==========================================

    transform = transforms.Compose([
        # transforms.Resize((512, 512)),    
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    bpp = []
    images = glob.glob(args.img_path + '/*.png')
    print(f'\nFind {str(len(images))} images in {args.img_path}\n')

    result = {}
    per_image_results = []  # Detailed per-image metrics for saving

    # Performance tracking variables
    perf_stats = []  # List to store per-image performance stats
    torch.cuda.reset_peak_memory_stats(device)  # Reset peak memory stats
    
    for img_idx, img_path in enumerate(images):
        (path, name) = os.path.split(img_path)
        fname, ext = os.path.splitext(name)
        outf = os.path.join(args.rec_path, fname+'.png')

        img = preprocess_image(img_path, transform).cuda().unsqueeze(0)
        ori_h, ori_w = img.shape[2:]

        pad_h = (math.ceil(ori_h / 256)) * 256 - ori_h
        pad_w = (math.ceil(ori_w / 256)) * 256 - ori_w
        img_padded = F.pad(img, pad=(0, pad_w, 0, pad_h), mode='reflect') # [-1,1]

        # Start timing and memory tracking
        torch.cuda.synchronize()
        start_time = time.time()
        mem_before = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
        
        with torch.no_grad():
            if args.entropy_estimation:
                N, _, H, W = img.size()
                num_pixels = N * H * W
                out_img, rate = net(img_padded, args.quality_index)
                
                rate = sum(
                    (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                    for likelihoods in rate.values()
                )
                out_img = out_img[:, :, 0 : ori_h, 0 : ori_w]
                out_img = (out_img * 0.5 + 0.5).float().detach()
            else:
                rate = compress_one_image(
                    net,
                    args.bin_path,
                    ori_h,
                    ori_w,
                    fname,
                    img_padded,
                    args.quality_index,
                )
                out_img = decompress_one_image(net, args.bin_path, ori_h, ori_w, fname)
        
        # End timing and memory tracking
        torch.cuda.synchronize()
        end_time = time.time()
        mem_after = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
        
        inference_time = end_time - start_time
        mem_used = mem_after - mem_before
        
        out_img = out_img.clamp(0.0, 1.0) # [0, 1]
        gt_img = (img * 0.5 + 0.5).float().detach()
        
        # compute metrics
        print(f'=============={fname}==============')

        # Record per-image data
        bpp_val = float(rate.cpu().numpy()) if hasattr(rate, 'cpu') else float(rate)
        per_image_record = {
            'image_name': fname,
            'resolution': f'{ori_w}x{ori_h}',
            'bpp': bpp_val,
            'inference_time_s': inference_time,
            'memory_used_mb': mem_used,
            'peak_memory_mb': peak_mem,
        }

        # Record performance stats
        perf_record = {
            'image_name': fname,
            'inference_time': inference_time,
            'memory_before_mb': mem_before,
            'memory_after_mb': mem_after,
            'memory_used_mb': mem_used,
            'peak_memory_mb': peak_mem,
            'bpp': bpp_val,
            'resolution': f'{ori_w}x{ori_h}'
        }
        perf_stats.append(perf_record)

        if args.verbose_perf:
            print(f"[Performance] Time: {inference_time:.4f}s | "
                  f"Mem Before: {mem_before:.2f}MB | Mem After: {mem_after:.2f}MB | "
                  f"Mem Used: {mem_used:.2f}MB | Peak: {peak_mem:.2f}MB")

        for key, metric in metric_dict.items():
            value = metric(out_img).item()
            print(key, value)
            result[key] = result.get(key, 0) + value
            per_image_record[key] = float(value)

        update_patch_fid(gt_img, out_img, fid_metric=fid_metric, kid_metric=kid_metric)

        for key, metric in metric_paired_dict.items():
            value = metric(out_img, gt_img).item()
            print(key, value)
            result[key] = result.get(key, 0) + value
            per_image_record[key] = float(value)

        try:
            bpp.append(rate.cpu().numpy())
        except:
            bpp.append(rate)
        print('[BPP]', rate)

        per_image_results.append(per_image_record)

        if args.save_img:
            output_pil = transforms.ToPILImage()(out_img[0].cpu()) 
            output_pil.save(outf)
        
        # Clear cache periodically to avoid memory fragmentation
        if (img_idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    print(f'==============The Avg. Results w/ {str(len(images))} images==============')
    result['fid'] = float(fid_metric.compute())

    if len(images) > 50:
        kid_tuple = kid_metric.compute()
        result['kid_mean'], result['kid_std'] = float(kid_tuple[0]), float(kid_tuple[1])

    # Build summary dict
    avg_bpp = float(np.mean(bpp))
    summary_metrics = {
        'num_images': len(images),
        'avg_bpp': avg_bpp,
        'results_dir': run_dir,
        'checkpoint': args.codec_path,
        'config': args.config_path,
        'use_ema': args.use_ema,
        'use_merge': args.use_merge,
        'entropy_estimation': args.entropy_estimation,
        'quality_index': args.quality_index,
        'timestamp': timestamp,
    }

    print_results = []
    for key, res in result.items():
        avg_val = res if (key in ('fid', 'kid_mean', 'kid_std')) else res / len(images)
        if key == 'fid':
            print(f"{key}: {avg_val:.2f}")
        elif key == 'kid_mean' or key == 'kid_std':
            print(f"{key}: {avg_val:.7f}")
        else:
            print(f"{key}: {avg_val:.5f}")
        summary_metrics[key] = float(avg_val)
        print_results.append(f"{key}: {avg_val:.5f}")

    print('\n[Average BPP]', avg_bpp)

    # Save per-image detailed results as JSON
    try:
        per_image_path = os.path.join(run_dir, "per_image_metrics.json")
        with open(per_image_path, 'w') as f:
            json.dump(per_image_results, f, indent=2, default=str)
        print(f'\nPer-image metrics saved to: {per_image_path}')
    except Exception as e:
        print(f'\nWarning: Could not save per-image metrics: {e}')

    # Save summary metrics as JSON
    try:
        summary_path = os.path.join(run_dir, "summary_metrics.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_metrics, f, indent=2, default=str)
        print(f'Summary metrics saved to: {summary_path}')
    except Exception as e:
        print(f'Warning: Could not save summary metrics: {e}')

    # Save per-image results as CSV (easy for spreadsheet analysis)
    try:
        if per_image_results:
            csv_path = os.path.join(run_dir, "per_image_metrics.csv")
            import csv
            all_keys = list(per_image_results[0].keys())
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=all_keys)
                writer.writeheader()
                writer.writerows(per_image_results)
            print(f'Per-image CSV saved to: {csv_path}')
    except Exception as e:
        print(f'Warning: Could not save CSV: {e}')

    # Print performance statistics
    if args.verbose_perf and len(perf_stats) > 0:
        print(f'\n{"="*60}')
        print(f'Performance Statistics ({len(perf_stats)} images)')
        print(f'{"="*60}')
        
        times = [p['inference_time'] for p in perf_stats]
        mem_used_list = [p['memory_used_mb'] for p in perf_stats]
        peak_mem_list = [p['peak_memory_mb'] for p in perf_stats]
        bpp_list = [p['bpp'] for p in perf_stats]
        
        print(f'\nInference Time:')
        print(f'  Average: {np.mean(times):.4f}s ± {np.std(times):.4f}s')
        print(f'  Min:     {np.min(times):.4f}s')
        print(f'  Max:     {np.max(times):.4f}s')
        print(f'  Median:  {np.median(times):.4f}s')
        print(f'  FPS:     {1.0/np.mean(times):.2f}')
        
        print(f'\nMemory Usage:')
        print(f'  Average Used: {np.mean(mem_used_list):.2f}MB ± {np.std(mem_used_list):.2f}MB')
        print(f'  Average Peak: {np.mean(peak_mem_list):.2f}MB ± {np.std(peak_mem_list):.2f}MB')
        print(f'  Max Peak:     {np.max(peak_mem_list):.2f}MB')
        
        print(f'\nBPP:')
        print(f'  Average: {np.mean(bpp_list):.4f} ± {np.std(bpp_list):.4f}')
        print(f'  Min:     {np.min(bpp_list):.4f}')
        print(f'  Max:     {np.max(bpp_list):.4f}')
        
        # Save detailed performance stats to file
        perf_output_path = os.path.join(run_dir, 'performance_stats.txt')
        try:
            with open(perf_output_path, 'w') as f:
                f.write(f'Performance Statistics for {len(perf_stats)} images\n')
                f.write(f'{"="*60}\n\n')
                
                f.write('Summary:\n')
                f.write(f'  Inference Time - Avg: {np.mean(times):.4f}s, Std: {np.std(times):.4f}s, '
                       f'Min: {np.min(times):.4f}s, Max: {np.max(times):.4f}s, FPS: {1.0/np.mean(times):.2f}\n')
                f.write(f'  Memory Used    - Avg: {np.mean(mem_used_list):.2f}MB, Std: {np.std(mem_used_list):.2f}MB, '
                       f'Max Peak: {np.max(peak_mem_list):.2f}MB\n')
                f.write(f'  BPP            - Avg: {np.mean(bpp_list):.4f}, Std: {np.std(bpp_list):.4f}\n\n')
                
                f.write('Detailed Per-Image Stats:\n')
                f.write(f'{"Image Name":<30} {"Resolution":<15} {"Time(s)":<12} {"Mem Used(MB)":<15} {"Peak Mem(MB)":<15} {"BPP":<10}\n')
                f.write('-' * 97 + '\n')
                for p in perf_stats:
                    f.write(f'{p["image_name"]:<30} {p["resolution"]:<15} {p["inference_time"]:<12.4f} '
                           f'{p["memory_used_mb"]:<15.2f} {p["peak_memory_mb"]:<15.2f} {p["bpp"]:<10.4f}\n')
            
            print(f'\nDetailed performance stats saved to: {perf_output_path}')
        except Exception as e:
            print(f'\nWarning: Could not save performance stats: {e}')
    
    # ===== 新增：对比测试 - 加载文本编码器后的推理速度 =====
    if args.measure_text_encoder and args.text_encoder_path and len(images) > 0:
        print(f'\n\n{"="*60}')
        print(f'TEXT ENCODER IMPACT TEST')
        print(f'{"="*60}')
        print(f'\nNow loading text encoder and running inference again for comparison...\n')
        
        # Load text encoder (keep it in memory)
        text_encoder, te_stats = measure_text_encoder_memory(
            args.text_encoder_path, 
            device, 
            continue_inference=True
        )
        
        if text_encoder is not None:
            # Measure memory with text encoder loaded
            torch.cuda.synchronize()
            mem_with_te = torch.cuda.memory_allocated(device) / (1024 ** 2)
            print(f'\nMemory with text encoder loaded: {mem_with_te:.2f} MB')
            print(f'Text encoder memory: {te_stats["memory_mb"]:.2f} MB\n')
            
            # Run inference again with text encoder in memory
            print(f'Running inference with text encoder loaded ({len(images)} images)...\n')
            
            perf_stats_with_te = []
            torch.cuda.reset_peak_memory_stats(device)
            
            # Use a subset of images for faster comparison (first 10 images or all if less)
            test_images = images[:min(10, len(images))]
            
            for img_idx, img_path in enumerate(test_images):
                (path, name) = os.path.split(img_path)
                fname, ext = os.path.splitext(name)
                
                img = preprocess_image(img_path, transform).cuda().unsqueeze(0)
                ori_h, ori_w = img.shape[2:]
                
                pad_h = (math.ceil(ori_h / 256)) * 256 - ori_h
                pad_w = (math.ceil(ori_w / 256)) * 256 - ori_w
                img_padded = F.pad(img, pad=(0, pad_w, 0, pad_h), mode='reflect')
                
                # Start timing
                torch.cuda.synchronize()
                start_time = time.time()
                
                with torch.no_grad():
                    if args.entropy_estimation:
                        N, _, H, W = img.size()
                        num_pixels = N * H * W
                        out_img, rate = net(img_padded, args.quality_index)
                    else:
                        rate = compress_one_image(
                            net,
                            args.bin_path,
                            ori_h,
                            ori_w,
                            fname,
                            img_padded,
                            args.quality_index,
                        )
                        out_img = decompress_one_image(net, args.bin_path, ori_h, ori_w, fname)
                
                # End timing
                torch.cuda.synchronize()
                end_time = time.time()
                inference_time = end_time - start_time
                
                perf_stats_with_te.append({
                    'image_name': fname,
                    'inference_time': inference_time,
                })
                
                if args.verbose_perf:
                    print(f'  [{img_idx+1}/{len(test_images)}] {fname}: {inference_time:.4f}s')
                
                # Clean up
                del img, img_padded, out_img
                torch.cuda.empty_cache()
            
            # Calculate statistics with text encoder
            times_with_te = [p['inference_time'] for p in perf_stats_with_te]
            
            print(f'\n{"="*60}')
            print(f'COMPARISON RESULTS')
            print(f'{"="*60}')
            print(f'\nWithout Text Encoder (baseline):')
            print(f'  Average Time: {np.mean(times):.4f}s')
            print(f'  FPS:          {1.0/np.mean(times):.2f}')
            
            print(f'\nWith Text Encoder Loaded (but not used):')
            print(f'  Average Time: {np.mean(times_with_te):.4f}s')
            print(f'  FPS:          {1.0/np.mean(times_with_te):.2f}')
            
            # Calculate speed difference
            time_diff = np.mean(times_with_te) - np.mean(times)
            speed_ratio = np.mean(times) / np.mean(times_with_te)
            slowdown_pct = (time_diff / np.mean(times)) * 100
            
            print(f'\nImpact Analysis:')
            print(f'  Time Difference:  {time_diff:+.4f}s ({slowdown_pct:+.2f}%)')
            print(f'  Speed Ratio:      {speed_ratio:.3f}x')
            print(f'  Text Encoder Mem: {te_stats["memory_mb"]:.2f} MB')
            
            if abs(slowdown_pct) < 2.0:
                print(f'\n✓ Text encoder has MINIMAL impact on inference speed (< 2%)')
            elif slowdown_pct > 0:
                print(f'\n⚠ Text encoder slows down inference by {slowdown_pct:.2f}%')
            else:
                print(f'\n✓ Text encoder actually speeds up inference by {-slowdown_pct:.2f}% (unexpected)')
            
            # Save comparison to file
            comparison_path = os.path.join(run_dir, 'text_encoder_impact_comparison.txt')
            try:
                with open(comparison_path, 'w') as f:
                    f.write(f'Text Encoder Impact on Inference Speed\n')
                    f.write(f'{"="*60}\n\n')
                    f.write(f'Without Text Encoder:\n')
                    f.write(f'  Average Time: {np.mean(times):.4f}s\n')
                    f.write(f'  FPS:          {1.0/np.mean(times):.2f}\n\n')
                    f.write(f'With Text Encoder Loaded:\n')
                    f.write(f'  Average Time: {np.mean(times_with_te):.4f}s\n')
                    f.write(f'  FPS:          {1.0/np.mean(times_with_te):.2f}\n')
                    f.write(f'  Memory Used:  {te_stats["memory_mb"]:.2f} MB\n\n')
                    f.write(f'Impact:\n')
                    f.write(f'  Time Difference:  {time_diff:+.4f}s ({slowdown_pct:+.2f}%)\n')
                    f.write(f'  Speed Ratio:      {speed_ratio:.3f}x\n')
                print(f'\nComparison report saved to: {comparison_path}')
            except Exception as e:
                print(f'\nWarning: Could not save comparison report: {e}')
            
            # Cleanup text encoder
            print(f'\nCleaning up text encoder...')
            del text_encoder
            torch.cuda.empty_cache()
            gc.collect()
            print(f'Done!\n')

    # Close log file
    print(f'\n[{"="*60}]')
    print(f'Inference completed. All results saved to: {run_dir}')
    print(f'[{"="*60}]')
    log_tee.close()


if __name__ == "__main__":
    args = parse_args_testing()
    if args.save_img:
        if not os.path.exists(args.rec_path): os.makedirs(args.rec_path)
        if not os.path.exists(args.bin_path): os.makedirs(args.bin_path)
    # Always ensure results dir exists
    if not os.path.exists(args.results_dir): os.makedirs(args.results_dir)
    main(args)
