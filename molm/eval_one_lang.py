import os
from argparse import ArgumentParser
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from molm.humaneval_qwen2_base import HumanEval as MultiPLERunner

DATA_ROOT = str(Path(__file__).joinpath("../data").resolve())
print(f"{DATA_ROOT = }")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument("--modelpath", type=str, default="")
    parser.add_argument("--language", type=str, default="")
    parser.add_argument("--no_batching", action="store_true", default=False)
    # Add new arguments specific to transformers if needed
    parser.add_argument("--device_map", type=str, default="auto", 
                       help="Device map strategy for model loading")
    parser.add_argument("--torch_dtype", type=str, default="auto",
                       help="torch dtype for model loading (auto, float16, bfloat16)")
    
    args = parser.parse_args()
    
    logdir = args.logdir
    if logdir == "":
        logdir = "debug/"
    Path(logdir).mkdir(exist_ok=True, parents=True)
    
    # Set up evaluator
    evaluator = MultiPLERunner(
        data_root=DATA_ROOT,
        max_seq_len=4096,
        log_dir=logdir,
        n_sample=1,
        language=args.language,
        max_gen_len=500,
        no_batching=args.no_batching,
    )
    
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    
    # Configure torch dtype
    torch_dtype = args.torch_dtype
    if torch_dtype != "auto":
        torch_dtype = getattr(torch, torch_dtype)
    
    # Load model and tokenizer
    print(f"Loading model from {args.modelpath}")
    model = AutoModelForCausalLM.from_pretrained(
        args.modelpath,
        device_map=args.device_map,  # "auto" will handle multi-GPU automatically
        torch_dtype=torch_dtype,
        trust_remote_code=True  # needed for some models
    )
    
    # Set tokenizer parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Run evaluation
    evaluator.eval_model(args.modelpath)  # Pass model path instead of model instance