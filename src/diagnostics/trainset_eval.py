import os, sys
os.chdir("/home/huyx/study/projects/llm-align-journey/src")
sys.path.insert(0, os.getcwd())
import gsm8k_eval
gsm8k_eval.GOLD_JSONL = "data/gsm8k_sft_train.jsonl"
print("=== 决定性测试：r8_merged 评训练集前300条 ===", flush=True)
acc = gsm8k_eval.do_eval("exp/sft_full_r8_merged", 300)
print("TRAINSET_ACC:", acc, flush=True)
