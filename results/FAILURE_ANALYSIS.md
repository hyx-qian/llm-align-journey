# 失败案例分析：10 条（badcase 归因 + 训练动态 + 工程事故）

> 每条格式：**现象 → 归因 → 修复/规避**。样本级案例取自 `results/eval_results_*.jsonl`（可复现），
> 训练级与工程级对应各自日志与 `src/diagnostics/`。

## A 类：模型级 badcase（预测错误，6 条）

**① "half that much" 语义反转（naive SFT r8）**
- 现象：袍子题 "takes 2 bolts of blue fiber and **half that much** white fiber"，模型算 2×2=4（blue 的两倍），正确为 2 的一半=1（total 3）
- 归因：比较短语的语义 grounding 失败——"half that much" 指向 blue 的一半，模型解析为"blue 的两倍"
- 规避：reward 只判最终对错，无法直接惩罚理解错误——这正是 process supervision / 更强基座的动机

**② 条件映射错误（naive SFT r8，Janet 鸭子题）**
- 现象：模型把"吃 3 个后的剩余蛋 × $2"算成"13 × 2 = 26"，gold 是 9 × 2 = 18
- 归因：多步条件链中段丢失（吃掉的 3 个/朋友买的 7 个约束未全部生效）
- 规避：见④⑤——同类错误在 GRPO 后仍存在，说明 SFT 数据的推理深度不足以根治

**③ 百分比基数误读（naive SFT r8，Mark 花园题）**
- 现象："80% more of those in purple"（紫 = 黄 ×1.8）被算成 黄 ×0.8 的一部分
- 归因：百分比比较句式（more/less than）基数方向判定错误
- 规避：数据侧——构造含百分比比较句式的强化数据；模型侧——更大基座

**④ 价格条件张冠李戴（GRPO，Kylar 眼镜题）**
- 现象："15 × 5 + 1 × 3 = 78"，gold 为 24 + 40 = 64（8 个@$3 + 8 个@$5）
- 归因：两种单价与两种数量的配对关系错置——GRPO 后模型仍在犯，且错误形态与 SFT 期一致
- 规避：该类错误占比是 GRPO vs SFT 的对比指标之一（详见 RESULTS.md）

**⑤ 漏约束（GRPO，Carla 下载题）**
- 现象：80 + 20 + 20 = 120 分钟，gold 40 + 100 + 20 = 160——"Windows 更新 40 分钟"约束被跳过
- 归因：约束条件位于句中非显著位置，模型注意力跳读
- 规避：同②，也是 GRPO v2 难题课程（向"多约束题"倾斜）的选样依据

**⑥ 多条件优先级错误（GRPO，Wendi 鸡饲料题）**
- 现象：把"求还差多少"错误理解为"总量够不够"（输出 10，gold 20）
- 归因：题目目标句（问什么）与条件句的优先级混淆
- 规避：同上，进 v2 难题池

## B 类：训练级动态（4 条，非样本级）

**⑦ SFT 侵蚀本身（−7.58 系统性退化）**
- 现象：68.16% → 60.58%，r8/r16 一致，badcase 集中于漏约束/误读
- 归因：官方短 CoT（短、浅、单一风格）使强基座习得浅层模式模仿，丢失细致推理
- 修复：RFT 修复至 73.46%（见 RESULTS.md 发现 2）——"数据质量决定 SFT 成败"的完整闭环

**⑧ GRPO 熵下降（0.37 → 0.21，−43%）**
- 现象：1 epoch 内熵持续下降——策略变确定，探索减少
- 归因：on-policy 循环的自我强化（采样分布收窄 → 训练目标收窄）
- 规避：1 epoch 后主动刹车；v2 若续训需监控 entropy 作为刹车指标，或引入 epsilon_high（DAPO clip-higher）

**⑨ 数据饥饿（frac_reward_zero_std 0.40 → 0.525）**
- 现象：训练末期 53% 的组内 std=0（8 发全对或全错）——无学习信号
- 归因：模型变强 → 简单题固化全对、难课题持续全错，中间难度题占比萎缩
- 规避：难题课程（采样池向能力边界倾斜）+ 加大 G + 换更强采样源

**⑩ DPO likelihood displacement（chosen reward 转负）**
- 现象：DPO 冒烟中 rewards/chosen +0.05 → −0.50（logps/chosen −71 → −87），margin 仍为正
- 归因：DPO 只约束相对 margin，chosen 的绝对概率可下降——离线方法的结构性缺陷
- 规避：控制 epoch、监控 chosen logps；这也是转向 GRPO（on-policy）的动机之一

## C 类：工程级事故（4 条，简要——详见 src/diagnostics/ 与 RESULTS.md 协议注记）

- **merge_and_unload 权重损坏**：transformers 5.16.1 + peft 0.20.0 组合下合并产物失效
  （原生 69% vs 合并 44%）→ 决定性对照实验定位 → 评测改走 vLLM 原生 LoRA 路径，机制待查
- **评测协议错位**：tokenizer 5.16 新格式被 4.57 端读取失败 → 权重与 tokenizer 分离指定
- **vLLM 依赖地狱**：安装 vLLM 时 transformers 被自动降级致训练栈崩坏 → 训练/推理环境分离
  → GRPO colocate 场景的受控合并四步 SOP（装/卸弹/恢复/三项验证）
- **共享服务器资源事件**：sshd 假死（swap 风暴）/ OOM 杀进程 → 显存侦察 + 逐 epoch 存档
  + resume_from_checkpoint 自保流程

## 归因总结

- **A 类（语义理解）**：占错误的主体——RLVR 的 outcome reward 无法直接惩罚"理解错"，
  这是从 outcome supervision 走向 process supervision 的直接动机
- **B 类（训练动态）**：RLHF 家族共有的稳定性问题——熵/数据饥饿/displacement 的监控
  与刹车是可迁移的工程经验
- **C 类（工程）**：版本管理与环境纪律——所有事故均有对照实验定案与留档
