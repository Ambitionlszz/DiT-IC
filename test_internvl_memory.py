#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 InternVL 模型在训练时的显存占用

使用方法:
    # 快速测试（默认配置）
    python test_internvl_memory.py
    
    # 指定模型路径和 batch sizes
    python test_internvl_memory.py \
        --model_path /root/share/new_models/OpenGVLab/InternVL2-2B \
        --batch_sizes 1 2 4 8 \
        --image_size 224
    
    # 只测试推理（不测试反向传播）
    python test_internvl_memory.py --test_backward=False
    
    # 保存结果到文件
    python test_internvl_memory.py --save_results
"""

import torch
import time
import argparse
import gc
from pathlib import Path
import csv


def get_gpu_memory():
    """获取当前 GPU 显存占用 (MB)"""
    if not torch.cuda.is_available():
        return 0, 0, 0
    
    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    return allocated, reserved, total


def clear_gpu_memory():
    """清理 GPU 显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        time.sleep(1)


def load_internvl_model(model_path):
    """加载 InternVL 模型"""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    
    print(f"📥 加载模型：{model_path}")
    start = time.time()
    
    try:
        # 加载模型和 processor
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,  # 使用 bfloat16 节省显存
            device_map="cuda",
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        load_time = time.time() - start
        print(f"✅ 模型加载成功，耗时：{load_time:.2f}s")
        
        # 设置模型为训练模式
        model.train()
        
        # 计算模型参数量
        num_params = sum(p.numel() for p in model.parameters())
        print(f"📊 模型参数量：{num_params / 1e9:.2f}B")
        
        return model, processor
        
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        raise


def create_dummy_input(batch_size, image_size, processor):
    """创建虚拟的图像 - 文本输入"""
    from PIL import Image
    import numpy as np
    
    # 创建虚拟图像（随机噪声）
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
    )
    
    # 创建虚拟文本
    dummy_text = "Describe this image in detail."
    
    # 批量处理
    images = [dummy_image] * batch_size
    texts = [dummy_text] * batch_size
    
    # 使用 processor 处理
    inputs = processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding=True
    )
    
    # 移动到 GPU
    inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    return inputs


def test_forward_backward(model, inputs, batch_size):
    """测试前向传播和反向传播的显存和时间"""
    clear_gpu_memory()
    initial_alloc, _, total_mem = get_gpu_memory()
    
    # 记录初始显存
    peak_alloc = initial_alloc
    
    # ========== 前向传播 ==========
    start_fwd = time.time()
    
    try:
        outputs = model(**inputs)
        torch.cuda.synchronize()
        
        fwd_time = time.time() - start_fwd
        after_fwd_alloc, _, _ = get_gpu_memory()
        peak_alloc = max(peak_alloc, after_fwd_alloc)
        
        # 获取 loss（如果有 labels）
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss
        else:
            # 如果没有自动计算 loss，创建一个 dummy loss
            loss = outputs.logits.sum() * 0.001
        
    except Exception as e:
        print(f"❌ 前向传播失败：{e}")
        return None, None, None
    
    # ========== 反向传播 ==========
    start_bwd = time.time()
    
    try:
        loss.backward()
        torch.cuda.synchronize()
        
        bwd_time = time.time() - start_bwd
        after_bwd_alloc, _, _ = get_gpu_memory()
        peak_alloc = max(peak_alloc, after_bwd_alloc)
        
    except Exception as e:
        print(f"❌ 反向传播失败：{e}")
        bwd_time = 0
        after_bwd_alloc = peak_alloc
    
    # 清理 gradients
    model.zero_grad(set_to_none=True)
    
    return fwd_time, bwd_time, peak_alloc - initial_alloc


def test_batch_size(model, processor, batch_size, image_size, test_backward=True):
    """测试特定 batch size 下的显存占用"""
    print(f"\n{'='*80}")
    print(f"测试 Batch Size: {batch_size}, Image Size: {image_size}x{image_size}")
    print(f"{'='*80}")
    
    # 创建输入
    print("创建虚拟输入...")
    inputs = create_dummy_input(batch_size, image_size, processor)
    
    # 预热（跳过第一次运行的额外开销）
    print("预热...")
    with torch.no_grad():
        _ = model(**inputs)
        torch.cuda.synchronize()
    
    clear_gpu_memory()
    time.sleep(1)
    
    # 正式测试
    print("开始测试...")
    results = []
    
    # 测试 3 次取平均值
    for i in range(3):
        if test_backward:
            fwd_time, bwd_time, mem_delta = test_forward_backward(
                model, inputs, batch_size
            )
        else:
            fwd_time, bwd_time, mem_delta = test_forward_backward(
                model, inputs, batch_size
            )
            bwd_time = 0  # 如果不测试 backward，设为 0
        
        if fwd_time is not None:
            results.append({
                'fwd_time': fwd_time,
                'bwd_time': bwd_time,
                'mem_delta': mem_delta
            })
    
    # 计算统计信息
    if len(results) > 0:
        avg_fwd_time = sum(r['fwd_time'] for r in results) / len(results)
        avg_bwd_time = sum(r['bwd_time'] for r in results) / len(results) if test_backward else 0
        avg_mem_delta = sum(r['mem_delta'] for r in results) / len(results)
        
        _, reserved_after, total_mem = get_gpu_memory()
        
        print(f"\n✅ 测试结果 (Batch Size={batch_size}):")
        print(f"  ⏱️  前向传播时间：{avg_fwd_time:.3f}s")
        if test_backward:
            print(f"  ⏱️  反向传播时间：{avg_bwd_time:.3f}s")
        print(f"  💾  显存增量：{avg_mem_delta:.1f}MB")
        print(f"  📊  总显存占用：{reserved_after:.1f}MB / {total_mem:.1f}MB ({reserved_after/total_mem*100:.1f}%)")
        
        return {
            'batch_size': batch_size,
            'image_size': image_size,
            'fwd_time': avg_fwd_time,
            'bwd_time': avg_bwd_time,
            'memory_delta': avg_mem_delta,
            'memory_total': reserved_after,
            'memory_percent': reserved_after / total_mem * 100
        }
    else:
        print("❌ 测试失败")
        return None


def main(args):
    print("="*80)
    print("🚀 InternVL 训练显存测试")
    print("="*80)
    
    # 检查 GPU
    if not torch.cuda.is_available():
        print("❌ 未检测到 CUDA 设备")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    print(f"📊 GPU: {gpu_name}")
    print(f"💾 总显存：{total_mem:.1f}MB")
    print("="*80)
    
    # 加载模型
    model, processor = load_internvl_model(args.model_path)
    
    # 等待一下
    time.sleep(2)
    
    # 测试不同 batch sizes
    all_results = []
    
    for batch_size in args.batch_sizes:
        result = test_batch_size(
            model, processor, batch_size, args.image_size, args.test_backward
        )
        if result is not None:
            all_results.append(result)
        
        # 清理显存
        clear_gpu_memory()
        time.sleep(2)
    
    # ========== 结果汇总 ==========
    if len(all_results) > 0:
        print("\n" + "="*100)
        print("📊 最终结果汇总")
        print("="*100)
        
        print(f"{'Batch Size':<12} | {'Image Size':<12} | {'Fwd Time(s)':<12} | "
              f"{'Bwd Time(s)':<12} | {'Mem Delta(MB)':<15} | {'Total Mem(MB)':<15} | {'Usage(%)':<10}")
        print("-"*100)
        
        for r in all_results:
            bwd_str = f"{r['bwd_time']:.3f}" if args.test_backward else "N/A"
            print(f"{r['batch_size']:<12} | {r['image_size']}x{r['image_size']:<10} | "
                  f"{r['fwd_time']:<12.3f} | {bwd_str:<12} | {r['memory_delta']:<15.1f} | "
                  f"{r['memory_total']:<15.1f} | {r['memory_percent']:<10.1f}")
        
        print("="*100)
        
        # 分析和建议
        print(f"\n💡 分析建议:")
        
        # 找出最大可用 batch size
        max_bs = max(all_results, key=lambda x: x['memory_percent'])
        if max_bs['memory_percent'] < 80:
            print(f"  ✅ 所有测试的 batch size 都在安全范围内 (<80% 显存)")
        else:
            print(f"  ⚠️  Batch Size={max_bs['batch_size']} 时显存占用较高 ({max_bs['memory_percent']:.1f}%)")
        
        # 推荐训练 batch size
        safe_results = [r for r in all_results if r['memory_percent'] < 70]
        if len(safe_results) > 0:
            recommended = max(safe_results, key=lambda x: x['batch_size'])
            print(f"  🎯 推荐训练 batch size: {recommended['batch_size']} "
                  f"(显存占用 {recommended['memory_percent']:.1f}%)")
        else:
            print(f"  ⚠️  所有测试的 batch size 显存占用都超过 70%，建议使用 gradient accumulation")
        
        # 保存结果
        if args.save_results:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"internvl_memory_test_{timestamp}.csv"
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
            
            print(f"\n📄 结果已保存到：{output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='测试 InternVL 训练时的显存占用')
    
    parser.add_argument('--model_path', type=str, 
                        default='/root/share/new_models/OpenGVLab/InternVL2-2B',
                        help='InternVL 模型路径')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 2, 4],
                        help='要测试的 batch sizes')
    parser.add_argument('--image_size', type=int, default=224,
                        help='输入图像的分辨率')
    parser.add_argument('--test_backward', type=bool, default=True,
                        help='是否测试反向传播')
    parser.add_argument('--save_results', action='store_true',
                        help='保存结果到 CSV 文件')
    
    args = parser.parse_args()
    main(args)