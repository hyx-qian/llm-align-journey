# -*- coding: utf-8 -*-
"""vLLM client：对接 rollout_multiturn 的 llm_client 接口约定（可调用对象）。

接口契约（与 FakeLLM 一致）：
    client(messages: list[dict], sampling_params) -> str  (本轮 assistant 生成文本)
"""
from vllm import LLM, SamplingParams


class VLLMClient:
    def __init__(self, model_path: str, gpu_memory_utilization: float = 0.85, max_tokens: int = 512):
        self.llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=gpu_memory_utilization)
        self.max_tokens = max_tokens

    def __call__(self, messages: list, sampling_params=None) -> str:
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=self.max_tokens)
        # vLLM chat 接口直接吃 messages 列表（自动套 chat template）
        outputs = self.llm.chat([messages], sampling_params)
        return outputs[0].outputs[0].text
