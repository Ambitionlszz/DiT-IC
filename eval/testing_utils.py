import argparse

def parse_args_testing(input_args=None):

    parser = argparse.ArgumentParser()
    # testing setting
    parser.add_argument("--config_path", type=str, default='/data/Kodak/')
    parser.add_argument("--codec_path", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU device id to use. Default: 0")
    # testing images
    parser.add_argument("--img_path", type=str, default='/data/Kodak/')

    # output path
    parser.add_argument("--rec_path", type=str, default='/output/rec/')
    parser.add_argument("--bin_path", type=str, default='/output/bin/')

    # testing details
    parser.add_argument("--seed", type=int, default=903, help="A seed for reproducible training.")
    parser.add_argument("--use_ema", action="store_true",)
    parser.add_argument("--save_img", action="store_true",)
    parser.add_argument("--use_merge", action="store_true",)
    parser.add_argument("--entropy_estimation", action="store_true",)
    
    # results output
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory to save inference results and logs. Default: results/")

    # performance monitoring
    parser.add_argument("--verbose_perf", action="store_true",
                       help="Enable verbose performance monitoring (inference time and memory usage per image)")
    
    # text encoder memory measurement
    parser.add_argument("--measure_text_encoder", action="store_true",
                       help="Measure text encoder memory usage (load but don't use)")
    parser.add_argument("--text_encoder_path", type=str, default=None,
                       help="Path to text encoder (e.g., ./SANA or HuggingFace model ID)")
    parser.add_argument("--continue_inference", action="store_true",
                       help="Continue with normal inference after measuring text encoder")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args
