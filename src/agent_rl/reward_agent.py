"""Agent RL v1 的 reward 组合——★★★ 由你实现（reward 设计是你的核心能力展示）。

要求：
  final_score = 可验证判对（0/1）+ 工具调用奖励（你设计权重与规则）
设计思考点（写注释，面试会问）：
  1. 工具调用是否应该"用了就给分"？——不，那会催生"无脑调用"的 hacking；
     更好的设计是什么？（提示：调用有效性 vs 调用次数）
  2. 多轮轨迹中，中途答案错误的轮次要惩罚吗？
  3. 权重 0.1 延续 v2 的 Format reward 设计还是调整？为什么？
"""
from rollout_multiturn import BOXED_RE
import re
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from gsm8k_eval import score  # 复用你 v2 的判分器（协议一致）


def combined_reward(trajectory_result: dict, gold_answer: str) -> float:
    """给定一次多轮 rollout 的结果与标准答案，返回组合 reward。

    Args:
        trajectory_result: rollout_multiturn 的返回值
            （含 trajectory / final_answer / num_tool_calls / turns_used）
        gold_answer: 标准答案文本（含 \\boxed 内容的原始 gold）
    Returns:
        float: 组合 reward
    """
    # ========== TODO 3（学员实现）：reward 组合 ==========
    # 提示：
    #   1. 判对分：final_answer 提取后与 gold 比对——复用 gsm8k_eval.score()
    #      注意 gold_answer 是官方 CoT 文本，需要先抽出 \\boxed 里的数字再比对
    #      （参考 eval_grpo_v2.py 的 gold 处理方式，协议一致！）
    #   2. 工具分：你设计——但先在注释里回答上面三个设计思考点，再写实现
    #   3. 返回范围建议 [0, 1.1]：判对 1.0 + 工具最多 0.1（与 v2 的 Format 设计同构）
    #reward设计思考：
    #1.工具调用了就给分吗：不，这样模型会学会调用工具无脑刷分，对于简单题也调用无疑背离初衷，应该按有效的调用比例给分
    #2.中途答案错误是否惩罚：不惩罚，否则容易减小模型探索性，最终答案错误奖励为0够重
    #3.是否使用0.1作为工具调用分上限：是，工具分不应过大或过小，且与之前format格式对比实验时保持相同维度
    # raise NotImplementedError("TODO 3(学员): reward 组合设计")
    
    correct = score(trajectory_result['final_answer'], gold_answer)
    valid = len([content for role, content in trajectory_result['trajectory'] if role=='tool' and 'ERROR' not in content])
    valid_ratio = valid / max(trajectory_result['num_tool_calls'], 1)
    tool_bonus = (valid_ratio * 0.1) if correct > 0 else 0.0
    return correct + tool_bonus
    
