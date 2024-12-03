import json
import os
from pathlib import Path
from human_eval.evaluation import evaluate_functional_correctness
from tqdm import tqdm
from molm.utils.dataset import HumanEvalDataset
from molm.utils.utils import cleanup_code
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

COMMON_EOS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
]

class HumanEval:
    """
    HumanEval evaluation class using HuggingFace transformers.
    """
    def __init__(
        self,
        data_root,
        max_seq_len=2048,
        language="python",
        max_gen_len=200,
        log_dir=None,
        temperature=0,
        top_p=0.95,
        n_sample=40,
        k_sample=1,
        no_batching=False,
    ):
        self.data_root = data_root
        self.max_seq_len = max_seq_len
        self.max_gen_len = max_gen_len
        self.k = k_sample
        self.n_sample = n_sample
        self.language = language
        self.log_dir = log_dir
        self.temperature = temperature
        self.top_p = top_p
        self.sft = False
        self.no_batching = no_batching
        self.eos = COMMON_EOS
        print(f"EOS: {self.eos}")
        os.makedirs(self.log_dir, exist_ok=True)
        
    def eval_model(self, model_name_or_path: str):
        assert self.log_dir is not None, "log_dir should not be None when evaluating humaneval"
        
        # Load model and tokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        dataset = HumanEvalDataset(self.data_root, sample_num=self.n_sample, language=self.language, issft=self.sft)
        if self.k > 1:
            assert self.n_sample >= 100, "HumanEval PASS@100 needs n_sample >= 100"

        # Generate
        with Path(self.log_file_path).open("w") as f_log:
            prompts = [data["prompt"] + "\n" for data in dataset]
            
            for idx, prompt in enumerate(tqdm(prompts[:2])):
                # Tokenize input
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, 
                                 max_length=self.max_seq_len).to(device)
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.max_gen_len,
                        do_sample=self.temperature > 0,
                        temperature=self.temperature if self.temperature > 0 else 1.0,
                        top_p=self.top_p,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=[tokenizer.convert_tokens_to_ids(eos) for eos in self.eos 
                                    if eos in tokenizer.vocab],
                        num_return_sequences=1
                    )
                
                # Decode output
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                suffixprediction = generated_text[len(prompt):]  # Remove the prompt from the beginning
                
                data = dataset[idx]
                prediction = prompt + suffixprediction
                suffixprediction = cleanup_code(suffixprediction, self.language, "humaneval", 
                                             self.sft, dataset.stopwords)
                original_prompt = data["original_prompt"]
                
                if not self.sft:
                    suffixprediction = original_prompt + "\n" + suffixprediction
                
                res = {
                    "task_id": data["task_id"],
                    "completion": suffixprediction,
                    "prompt": original_prompt,
                    "wholecode": prediction,
                }
                f_log.write(json.dumps(res, ensure_ascii=False) + "\n")

        # Aggregate scores
        self._calculate_final_score()

    @property
    def log_file_path(self) -> str:
        return os.path.join(self.log_dir, f"pred_{self.language}_output.jsonl")

    def _calculate_final_score(self):
        timeout = 10
        res, details = evaluate_functional_correctness(
            sample_file=self.log_file_path,
            problem_file=os.path.join(self.data_root, f"humaneval-{self.language}.jsonl"),
            timeout=timeout,
        )
        print(f"{self.language} score is", res["pass@%d" % self.k])
        details_file = os.path.join(self.log_dir, f"humaneval-{self.language}-details.json")
        with Path(details_file).open("w") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)
        print(f"Details => {details_file}")