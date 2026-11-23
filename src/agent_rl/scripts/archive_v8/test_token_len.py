from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
import json
lens = []
for line in open("data/gsm8k_coldstart.jsonl"):
    ex = json.loads(line)
    full = "".join(m["content"] for m in ex["prompt"]) + ex["completion"][0]["content"]
    lens.append(len(tok(full).input_ids))
lens.sort()
print(f"min={lens[0]}  median={lens[len(lens)//2]}  p95={lens[int(len(lens)*0.95)]}  max={lens[-1]}")
