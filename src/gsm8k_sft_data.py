from datasets import load_dataset
import json
import re
ds = load_dataset("openai/gsm8k", "main")   # train 7473 条 / test 1319 条

# print(ds)
def convert(example):
    # 1. GSM8K 的 answer 字段长这样: "推理过程...\n#### 42"
    #    用 "####" 切开 → CoT 和最终答案
    # 2. 清洗: 答案里的千分位逗号去掉（"1,000"→"1000"，否则评测时抽不出来）
    # 3. 组装成 conversational prompt-completion:
    #    {"prompt":     [{"role": "user",      "content": 题目}],
    #     "completion": [{"role": "assistant", "content": CoT + "\n\\boxed{答案}"}]}
    CoT, answer = re.sub(r"(?<=\d),(?=\d)", '', example['answer']).split('####')
    CoT = re.sub(r"<<[^>]*>>", "", CoT)
    answer = answer.strip()
    return {
        "prompt": [{'role': 'user', 'content': example['question']}],
        "completion": [{"role": "assistant", "content": f"{CoT}\n\\boxed{{{answer}}}"}]
    }

def dump_jsonl(ds, path):
    with open(path, 'w') as f:
        for example in ds:
            f.write(json.dumps(convert(example), ensure_ascii=False) + "\n")
            
dump_jsonl(ds["train"], "data/gsm8k_sft_train.jsonl")
dump_jsonl(ds["test"],  "data/gsm8k_sft_test.jsonl")

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    sample = convert(ds["train"][0])
    print(tok.apply_chat_template(sample["prompt"] + sample["completion"], tokenize=False))
