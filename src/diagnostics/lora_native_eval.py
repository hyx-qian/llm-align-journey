import os, sys
os.chdir("/home/huyx/study/projects/llm-align-journey/src")
sys.path.insert(0, os.getcwd())
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import gsm8k_eval

ds = load_dataset("json", data_files="data/gsm8k_sft_train.jsonl", split="train").select(range(300))
conversations = [[ex["prompt"][0]] for ex in ds]
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", dtype="bfloat16",
          gpu_memory_utilization=0.85, enable_lora=True, max_lora_rank=16)
sp = SamplingParams(temperature=0, max_tokens=512)
outputs = llm.chat(conversations, sp,
                   lora_request=LoRARequest("r8", 1, "exp/sft_full_r8/checkpoint-702"))
correct = 0
for ex, o in zip(ds, outputs):
    ok = gsm8k_eval.score(o.outputs[0].text, ex["completion"][0]["content"])
    correct += int(ok)
print(f"LORA_NATIVE_TRAIN_ACC: {correct/len(ds):.4f} ({correct}/{len(ds)})", flush=True)
