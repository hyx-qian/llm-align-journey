# -*- coding: utf-8 -*-
"""B/C 文档路线评测：模型不训练，只注入标准工具文档（TOOL_DOC_V1）。

对比矩阵：
  B = 1.5B base + 文档（本脚本 with --model qwen1.5b）
  C = 7B base   + 文档（本脚本 with --model qwen2.5-7b）
  （同文档文本、同评测集、同协议——只换模型）

评测集：
  E_old  = 旧工具题（calc 可解，来自 gsm8k_agent_train_4k 尾部 50）
  E_new  = 新工具题（需 python，data/e_new_python_tasks.jsonl）
  E_mix  = 混合题（data/e_mix_overlap_tasks.jsonl，可选）

指标：format_rate / exec_rate / EM / ans=None / tool_usage 分布 / stop_reason
成本：avg_input_tokens / avg_output_tokens（用于成本-效果对比）
"""
import argparse
import json
import os
import statistics
import time
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm import LLM, SamplingParams
from env_rollout import rollout_multiturn
from env_tool_doc import TOOL_DOC_CALC, TOOL_DOC_T2, TOOL_DOC_T3, TOOL_DOC_V1
from gsm8k_eval import score

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DATA = os.path.join(HERE, "data", "gsm8k_agent_train_4k.jsonl")
E_NEW_DATA = os.path.join(HERE, "data", "e_new_python_tasks.jsonl")
E_MIX_DATA = os.path.join(HERE, "data", "e_mix_overlap_tasks.jsonl")


class DocClient:
    """最小 vLLM chat 客户端（B/C 用，不挂 LoRA）。"""

    def __init__(self, model: str, gpu_mem: float = 0.85):
        self.llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=gpu_mem)
        self.tokenizer = self.llm.get_tokenizer()

    def __call__(self, messages: list, sampling_params) -> str:
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        out = self.llm.generate([text], sampling_params)[0]
        return out.outputs[0].text


def load_eval_set(name: str, n: int):
    """按名字读评测集文件（返回 [(question, gold_answer)]）。"""
    mapping = {
        "E_old_T1": "e_old_T1_200.jsonl",
        "E_old_T2": "e_old_T1_200.jsonl",
        "E_old_T3": "e_old_T3_250.jsonl",
        "E_new_T2": "e_new_python_tasks.jsonl",
        "E_new_T3": "e_new_T3_sympy.jsonl",
        "E_mix_T2": "e_mix_T2_collab.jsonl",
        "E_mix_T3": "e_mix_T3_3tool.jsonl",
    }
    if name not in mapping:
        raise ValueError(f"unknown eval set {name}")
    path = os.path.join(HERE, "data", mapping[name])
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [(r["question"], r["gold_answer"]) for r in rows][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="模型路径（1.5B base 或 7B base）")
    ap.add_argument("--eval", default="E_old_T1", choices=["E_old_T1","E_old_T2","E_old_T3","E_new_T2","E_new_T3","E_mix_T2","E_mix_T3"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--doc", default="t3", choices=["calc", "t2", "t3", "sys"],
                    help="calc=TOOL_DOC_CALC / t2=calc+python / t3=三工具 / sys=训练同款TOOL_SYSTEM_PROMPT")
    ap.add_argument("--gpu-mem", type=float, default=0.35)
    args = ap.parse_args()

    from env_rollout import TOOL_SYSTEM_PROMPT as _SYS
    system_prompt = {"calc": TOOL_DOC_CALC, "t2": TOOL_DOC_T2, "t3": TOOL_DOC_T3, "sys": _SYS}[args.doc]
    client = DocClient(args.model, args.gpu_mem)
    sp = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=512, stop=["</tool_call>"])

    data = load_eval_set(args.eval, args.n)
    doc_tokens = len(client.tokenizer(system_prompt)["input_ids"])
    results = []
    n_fmt = n_exec = n_ok = n_none = 0
    in_toks, out_toks = [], []
    reasons, usage = {}, {}

    for i, (q, gold) in enumerate(data):
        _t0 = time.time()
        tr = rollout_multiturn(client, q, sp, system_prompt=system_prompt)
        _dt_ms = (time.time() - _t0) * 1000
        traj_text = " ".join(c for _, c in tr["trajectory"])
        has_fmt = "<tool_call>" in traj_text
        has_exec = tr["num_tool_calls"] > 0
        em = score(tr["final_answer"], gold) if tr["final_answer"] else 0.0
        n_fmt += int(has_fmt); n_exec += int(has_exec); n_ok += int(em >= 1.0)
        n_none += int(tr["final_answer"] is None)
        reasons[tr["stop_reason"]] = reasons.get(tr["stop_reason"], 0) + 1
        for k, v in tr["tool_usage"].items():
            usage[k] = usage.get(k, 0) + v
        # 成本（token 计）
        in_toks.append(doc_tokens + len(client.tokenizer(q)["input_ids"]))
        out_toks.append(sum(len(client.tokenizer(c)["input_ids"]) for r, c in tr["trajectory"] if r == "assistant"))
        results.append({"idx": i, "q": q[:80], "turns": tr["turns_used"], "calls": tr["num_tool_calls"],
                        "usage": tr["tool_usage"], "ans": tr["final_answer"], "gold": gold[:40],
                        "em": em, "format": has_fmt, "exec": has_exec, "stop_reason": tr["stop_reason"],
                        "in_tok": in_toks[-1], "out_tok": out_toks[-1], "latency_ms": _dt_ms})
        print(f"[{i}] calls={tr['num_tool_calls']}{tr['tool_usage']} fmt={has_fmt} exec={has_exec} "
              f"em={em:.0f} reason={tr['stop_reason']} ans={tr['final_answer']}")

    n = len(results)
    print("=" * 60)
    print(f"model={os.path.basename(args.model)} | eval={args.eval} | n={n}")
    print(f"format_rate: {n_fmt/n*100:.0f}% | exec_rate: {n_exec/n*100:.0f}% | EM: {n_ok/n*100:.0f}% | ans=None: {n_none/n*100:.0f}%")
    print(f"tool_usage: {usage} | stop_reason: {reasons}")
    lat = [_r.get("latency_ms", 0) for _r in results]
    print(f"成本(每題): 输入 {statistics.mean(in_toks):.0f} tokens（含文档 {doc_tokens}） | 输出 {statistics.mean(out_toks):.0f} tokens | 耗时 {statistics.mean(lat)/1000:.1f}s")
    out_path = args.out or f"results/doc_{args.doc}_{os.path.basename(args.model)}_{args.eval}.jsonl"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
