import torch
import time
import gc
import argparse
import os

def measure_text_encoder_memory(model_path, device='cuda', output_file=None):
    """
    测量文本编码器的显存占用（加载但不使用）
    model_path: 应该指向包含 model_index.json 的模型根目录（如 ./SANA）
                而不是 text_encoder 子文件夹
    """
    print(f"\n{'='*60}")
    print(f"Measuring Text Encoder Memory Usage")
    print(f"{'='*60}")
    print(f"Model path: {model_path}")
    print(f"Device: {device}\n")
    
    # 检查路径
    if not os.path.exists(model_path):
        print(f"❌ Error: Path does not exist: {model_path}")
        return None
    
    # 检查是否错误地指向了 text_encoder 子文件夹
    if os.path.exists(os.path.join(model_path, 'config.json')) and not os.path.exists(os.path.join(model_path, 'model_index.json')):
        print(f"⚠️  Warning: The path seems to point to a subcomponent (like text_encoder/).")
        print(f"   Please provide the path to the model root directory containing 'model_index.json'.")
        print(f"   Example: ./SANA (not ./SANA/text_encoder)")
        # 尝试自动修正：向上一级
        parent_path = os.path.dirname(model_path)
        if os.path.exists(os.path.join(parent_path, 'model_index.json')):
            print(f"\n✓ Auto-correcting to parent directory: {parent_path}\n")
            model_path = parent_path
        else:
            print(f"\n❌ Cannot auto-correct. Please specify the correct model root path.\n")
            return None
    
    if not os.path.exists(os.path.join(model_path, 'model_index.json')):
        print(f"❌ Error: model_index.json not found in {model_path}")
        print(f"   Please provide the path to the complete model directory.")
        return None
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)  # 等待显存稳定
    
    mem_before = torch.cuda.memory_allocated(device) / (1024 ** 2)
    print(f"Memory before loading text encoder: {mem_before:.2f} MB")
    
    # Load text encoder
    try:
        from diffusers import SanaPipeline
        
        print(f"\nLoading text encoder from: {model_path}")
        print("This may take a while...\n")
        
        # Record time
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Load only text encoder (使用 device_map 避免 cpu offload 错误)
        pipeline = SanaPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # 使用 FP32 测量真实显存
        )
        text_encoder = pipeline.text_encoder
        
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
        
        # Count layers
        total_modules = sum(1 for _ in text_encoder.modules())
        linear_layers = sum(1 for m in text_encoder.modules() if isinstance(m, torch.nn.Linear))
        layer_norms = sum(1 for m in text_encoder.modules() if isinstance(m, torch.nn.LayerNorm))
        
        print(f"{'='*60}")
        print(f"Text Encoder Memory Statistics:")
        print(f"{'='*60}")
        print(f"  Load time:          {load_time:.2f}s")
        print(f"  Memory before:      {mem_before:.2f} MB")
        print(f"  Memory after:       {mem_after:.2f} MB")
        print(f"  Text encoder used:  {text_encoder_mem:.2f} MB ⭐")
        print(f"  Peak memory:        {peak_mem:.2f} MB")
        print(f"  Parameters:         {num_params_m:.2f}M")
        print(f"  Model class:        {text_encoder.__class__.__name__}")
        print(f"  Total modules:      {total_modules}")
        print(f"  Linear layers:      {linear_layers}")
        print(f"  LayerNorm layers:   {layer_norms}")
        
        # Show model architecture summary
        print(f"\n{'='*60}")
        print(f"Model Architecture Summary:")
        print(f"{'='*60}")
        print(text_encoder)
        
        # Cleanup
        print(f"\n{'='*60}")
        print(f"Cleaning up...")
        del text_encoder
        del pipeline
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)
        
        mem_after_cleanup = torch.cuda.memory_allocated(device) / (1024 ** 2)
        print(f"Memory after cleanup: {mem_after_cleanup:.2f} MB")
        print(f"Memory reclaimed: {mem_after - mem_after_cleanup:.2f} MB")
        print(f"{'='*60}\n")
        
        # Save to file if requested
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(f"Text Encoder Memory Measurement Report\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"Model path: {model_path}\n")
                    f.write(f"Device: {device}\n\n")
                    f.write(f"Load time:          {load_time:.2f}s\n")
                    f.write(f"Memory before:      {mem_before:.2f} MB\n")
                    f.write(f"Memory after:       {mem_after:.2f} MB\n")
                    f.write(f"Text encoder used:  {text_encoder_mem:.2f} MB\n")
                    f.write(f"Peak memory:        {peak_mem:.2f} MB\n")
                    f.write(f"Parameters:         {num_params_m:.2f}M\n")
                    f.write(f"Model class:        {text_encoder.__class__.__name__}\n")
                    f.write(f"Total modules:      {total_modules}\n")
                    f.write(f"Linear layers:      {linear_layers}\n\n")
                    f.write(f"Memory after cleanup: {mem_after_cleanup:.2f} MB\n")
                    f.write(f"Memory reclaimed: {mem_after - mem_after_cleanup:.2f} MB\n")
                print(f"Report saved to: {output_file}\n")
            except Exception as e:
                print(f"Warning: Could not save report: {e}\n")
        
        return {
            'memory_mb': text_encoder_mem,
            'peak_memory_mb': peak_mem,
            'params_m': num_params_m,
            'load_time': load_time
        }
        
    except ImportError:
        print(f"\n❌ Error: diffusers not installed or SanaPipeline not available")
        print(f"Please install diffusers: pip install diffusers")
        return None
    except Exception as e:
        print(f"\n❌ Error loading text encoder: {e}")
        print(f"\n可能需要的操作:")
        print(f"  1. 确保已下载完整的 SANA 模型（包含 text_encoder 文件夹）")
        print(f"  2. 检查 diffusers 版本是否支持 SanaPipeline")
        print(f"  3. 确认模型路径正确: {model_path}")
        print(f"\n完整错误信息:")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Measure text encoder memory usage')
    parser.add_argument('--model_path', type=str, default='./SANA',
                       help='Path to SANA model (local directory or HuggingFace model ID)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file path for saving the report')
    
    args = parser.parse_args()
    
    measure_text_encoder_memory(args.model_path, args.device, args.output_file)


if __name__ == '__main__':
    main()
