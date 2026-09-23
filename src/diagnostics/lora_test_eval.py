import os, sys, json
os.chdir("/home/huyx/study/projects/llm-align-journey/src")
sys.path.insert(0, os.getcwd())
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import gsm8k_eval

ds = load_dataset("json", data_files="data/gsm8k_sft_test.jsonl", split="train")
conversations = [[ex["prompt"][0]] for ex in ds]
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", dtype="bfloat16",
          gpu_memory_utilization=0.85, enable_lora=True, max_lora_rank=16)
sp = SamplingParams(temperature=0, max_tokens=512)

for name, lora_id, path in [("r8", 1, "exp/sft_full_r8/checkpoint-702"),
                            ("r16", 2, "exp/sft_full_r16/checkpoint-702")]:
    outputs = llm.chat(conversations, sp, lora_request=LoRARequest(name, lora_id, path))
    correct, results = 0, []
    for ex, o in zip(ds, outputs):
        ok = gsm8k_eval.score(o.outputs[0].text, ex["completion"][0]["content"])
        correct += int(ok)
        results.append({"question": ex["prompt"][0]["content"][:80],
                        "pred": o.outputs[0].text[-200:],
                        "gold": ex["completion"][0]["content"][-80:], "correct": ok})
    with open(f"exp/eval_results_lora_{name}.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"TEST_ACC_{name}: {correct/len(ds):.4f} ({correct}/{len(ds)})", flush=True)
