from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import load_dataset

train_ds = load_dataset("json", data_files="data/gsm8k_sft_train.jsonl", split="train")

peft_config = LoraConfig(
    r=16,                              # TODO-A: 作业要求至少一组对比（r=8 vs 16，或 lr 两档）
    lora_alpha=32,                    # TODO-B: alpha 与 r 的关系？查一下通常 alpha=r 的几倍
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # TODO-C: 要不要加 gate/up/down？
    task_type="CAUSAL_LM",
)

cfg = SFTConfig(
    output_dir="exp/sft_full_r16",
    save_strategy="epoch",            # ← 加这行：每个 epoch 结束存一次 checkpoint
    per_device_train_batch_size=8,    # TODO-D: 5090 32G 显存上限内试，爆了就降或开 gradient_accumulation
    gradient_accumulation_steps=4,
    learning_rate=1e-4,               # 文档建议：PEFT 训 adapter 用 ~1e-4（比全参小两个量级）
    num_train_epochs=3,
    bf16=True,
    logging_steps=10,
    eos_token="<|im_end|>",           # ★ Qwen 必设，见 Q3
    completion_only_loss=True,        # 默认就是 True，显式写出来防手滑
    packing=False,                    # 先关，跑通后再做对比实验
)

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    args=cfg,
    train_dataset=train_ds,
    peft_config=peft_config,
)
trainer.train()