# -*- coding: utf-8 -*-
"""Agent RL 阶段 1 训练对接骨架：GRPO on 多轮工具调用轨迹。

★★★ 技术难点（先读再写）：
  TRL 1.12 的 GRPOTrainer 原生 rollout 是单轮的。多轮工具调用轨迹的训练对接有两种路径：

  路径 A（阶段 1 采用，简化）：把多轮轨迹（assistant/tool 交替）拼成一条
    conversational completion 喂 GRPOTrainer——工具轮的 token 不 mask（v1 简化），
    全部参与 loss。已知偏差：工具输出不是模型生成的，参与 loss 会引入轻微噪声——
    v1 接受这个偏差换管线简单，v4 再精细化 mask。

  路径 B（v4 精细化，先不做）：自定义 rollout + 工具轮 mask——需要改 GRPOTrainer
    的 loss 计算或用 completion_mask 参数（TRL 支持则用）。

  你的任务：
    1. 读 grpo_smoke_v2.py 的 GRPOTrainer 配置（复用超参：lr/beta/G=8/生成参数）；
    2. 写 GRPOConfig（对比 v2 的配置，哪些参数要为多轮调整：max_completion_length
       要变长——多轮轨迹更长；num_generations 仍 8）；
    3. 数据格式：确认 TRL 1.12 的 GRPOTrainer 吃 conversational 格式的方式
       （prompt 为 messages 列表，completion 由 trainer 在线生成——但你已经有
       rollout_multiturn 产出的离线轨迹，这里有个设计决策：在线 vs 离线，
       先按 GRPOTrainer 默认在线模式走，prompt 用 agent 数据的格式）；
    4. reward_funcs：把 reward_agent.combined_reward 包装成 TRL 的 reward 函数签名
       （TRL 的 reward 函数签名：def fn(completions, **kwargs) -> list[float]，
       需要适配：completions 是文本列表，gold 需要从 dataset 列传入）。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_training_args():
    """TODO(学员)：GRPOConfig——对比 grpo_smoke_v2.py 的配置，说明多轮场景的调整项。"""
    # 提示：
    #   from trl import GRPOConfig
    #   对比 v2：output_dir（agent_v1）、num_generations=8（不变）、
    #   max_completion_length（多轮轨迹更长，v2 是 512，这里建议 1024？说出你的理由）、
    #   learning_rate/beta 延续 v2（先不调，单变量原则）
    raise NotImplementedError("TODO(学员): GRPOConfig")


def build_reward_fn():
    """TODO(学员)：把 combined_reward 包装成 TRL reward 函数签名。

    TRL reward 函数签名：def fn(completions, prompts, **kwargs) -> list[float]
    难点：combined_reward 需要 gold_answer，而 TRL 的 reward 函数通过
      kwargs 拿 dataset 的额外列——确认 TRL 1.12 怎么传（dataset 的非 prompt 列
      会进 kwargs）——把 gold 列传进去。
    """
    raise NotImplementedError("TODO(学员): reward 函数适配")


def main():
    # TODO(学员)：组装 GRPOTrainer（数据 + config + reward_fn）→ train
    # 参考 grpo_smoke_v2.py 的主流程
    raise NotImplementedError("TODO(学员): 主流程")


if __name__ == "__main__":
    main()
