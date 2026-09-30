# dpo_smoke.py 核心配置
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig
from datasets import load_dataset
ds = load_dataset('json', data_files='data/gsm8k_dpo_k8.jsonl', split='train')
cfg = DPOConfig(
    output_dir="exp/dpo_smoke",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,     # 有效 batch 16 → 2067 对 ≈ 129 步/epoch，步数多点好看曲线
    learning_rate=5e-5,                # DPO 比 SFT 更怕大 lr（margin 崩），5e-5 起步
    beta=0.1,                          # 缰绳松紧，你手推过的那个 β
    num_train_epochs=2,
    bf16=True,
    logging_steps=5,
    # max_length=512,                  # 砍序列上限（GSM8K 渲染后 ~400 token 内，别让 pad 撑大 logits）
)
trainer = DPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    args=cfg,
    train_dataset=ds,                  # load_dataset("json", data_files="data/gsm8k_dpo.jsonl")
    peft_config=LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                           target_modules=["q_proj","k_proj","v_proj","o_proj"],
                           task_type="CAUSAL_LM"),
)
trainer.train()
