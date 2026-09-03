import torch
print("python path:", __import__('sys').executable)
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    x = torch.randn(2048, 2048, device=f"cuda:{i}")
    r = (x @ x).sum()
    print(f"GPU{i}: {torch.cuda.get_device_name(i)}",
          f"| capability={torch.cuda.get_device_capability(i)}",
          f"| matmul_ok={torch.isfinite(r).item()}")
