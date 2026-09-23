# -*- coding: utf-8 -*-
"""gsm8k_eval.py 骨架：SFT 前后准确率评测（W2 报告核心数字的来源）

三个阶段：
  A. merge：把 LoRA adapter 合并进基座 → 存出完整模型目录（vLLM 直接加载）
  B. 推理：vLLM 批量生成 test 集回答（temperature=0，可复现）
  C. 判分：\boxed{} 抽取 → 归一化 → 对比 gold → 输出 JSONL + accuracy

用法：
  python gsm8k_eval.py merge exp/sft_full_r8/checkpoint-702 exp/sft_full_r8_merged
  python gsm8k_eval.py eval exp/sft_full_r8_merged 200     # 先 200 条调通
  python gsm8k_eval.py eval Qwen/Qwen2.5-1.5B-Instruct 1319  # base 基线
"""
import json
import re
import sys

from datasets import load_dataset

MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
GOLD_JSONL = "data/gsm8k_sft_test.jsonl"      # 你自己生成的，gold 在 \boxed{} 里


# ---------- A. merge（给全） ----------
def do_merge(adapter_dir, out_dir):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(MODEL_PATH).save_pretrained(out_dir)  # vLLM 需要同目录 tokenizer
    print(f"merged -> {out_dir}")


# ---------- B. 推理（给全，注意用 llm.chat——它自动套 chat template） ----------
def do_infer(model_dir, n_samples):
    from vllm import LLM, SamplingParams

    ds = load_dataset("json", data_files=GOLD_JSONL, split="train").select(range(n_samples))
    # 把 prompt-completion 还原成 messages：只发问题，让模型自己生成回答
    conversations = [[m] for m in [ex["prompt"][0] for ex in ds]]   # [[{"role":"user",...}], ...]

    llm = LLM(model=model_dir, dtype="bfloat16", gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0, max_tokens=512)              # 贪心解码，结果可复现
    outputs = llm.chat(conversations, sp)

    preds = [o.outputs[0].text for o in outputs]
    golds = [ex["completion"][0]["content"] for ex in ds]
    return preds, golds


# ---------- C. 判分（TODO 在这里） ----------
def extract_answer(text):
    """TODO-1: 从文本里抽最终数字答案
    优先级：
      1) 找所有 r"\\boxed{...}"，取最后一个花括号内的内容（失败回答常把 \boxed 打错，取最后的兜底）
      2) 兜底：找不到 \boxed 就取文本中最后一个数字 r"-?\d[\d,]*"
    """
    res = re.findall(r'\\boxed\{(.*?)\}', text)   # (.*?) 捕获组：直接拿到花括号里的内容
    if res:                                        # 找到了 → 取最后一个
        return res[-1]
    res = re.findall(r'-?\d[\d,]*', text)          # 兜底：最后一个数字
    if res:
        return res[-1]
    return None
    # raise NotImplementedError("TODO-1")


def normalize(s):
    """TODO-2: 归一化——去千分位逗号、空格、$ 符号、结尾的 .0；转 float 失败就返回原字符串"""
    try:
        return float(re.sub(r'[,\s$]|\.0$', '', s))
    except (ValueError, TypeError):
        return s
    # raise NotImplementedError("TODO-2")


def score(pred_text, gold_text):
    pred = normalize(extract_answer(pred_text) or "")
    gold = normalize(extract_answer(gold_text) or "")
    try:                                   # 数字型：float 相等即对
        return abs(float(pred) - float(gold)) < 1e-6
    except (ValueError, TypeError):        # 非数字型：退回字符串全等
        return pred == gold


# ---------- 主入口（给全） ----------
def do_eval(model_dir, n_samples):
    from vllm import LLM, SamplingParams

    ds = load_dataset("json", data_files=GOLD_JSONL, split="train").select(range(n_samples))
    conversations = [[ex["prompt"][0]] for ex in ds]

    llm = LLM(model=model_dir, tokenizer=MODEL_PATH, dtype="bfloat16", gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0, max_tokens=512)
    outputs = llm.chat(conversations, sp)

    results, correct = [], 0
    for ex, o in zip(ds, outputs):
        pred_text = o.outputs[0].text
        gold_text = ex["completion"][0]["content"]
        ok = score(pred_text, gold_text)
        correct += int(ok)
        results.append({"question": ex["prompt"][0]["content"][:80],
                        "pred": pred_text[-200:], "gold": gold_text[-80:], "correct": ok})
        if not ok and len(results) <= 5:   # 前 5 条错题打出来看（failure case 的第一手素材）
            print("---- WRONG ----\nPRED:", pred_text[-300:], "\nGOLD:", gold_text[-120:])

    acc = correct / len(results)
    tag = model_dir.split("/")[-1]                     # r8_merged / r16_merged / Qwen2.5-1.5B-Instruct
    with open(f"exp/eval_results_{tag}.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"ACC = {acc:.4f}  ({correct}/{len(results)})  model={model_dir}")
    return acc


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "merge":
        do_merge(sys.argv[2], sys.argv[3])
    elif cmd == "eval":
        do_eval(sys.argv[2], int(sys.argv[3]))
