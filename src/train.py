# 四次迭代修改版
from my_attention import MultiHeadAttention
import torch
import torch.nn as nn
import torch.optim as optim

class Block(nn.Module):
            
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.re = nn.ReLU()
        self.fc2 = nn.Linear(4 * d_model, d_model)
        
    def forward(self, x, causal=True):
        x = x + self.attn(self.ln1(x), causal)
        x = x + self.fc2(self.re(self.fc1(self.ln2(x))))
        return x
            
class Model(nn.Module):
            
    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        self.norm_f = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm_f(x)
            
class Transformer:
    
    def __init__(self,  n_heads=4, n_layers=2, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.H = n_heads
        self.L = n_layers
        self.lr =lr
        
    def train(self, x, y, epochs=200):
        model = Model(x.shape[-1], self.H, self.L).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        x, y = x.to(self.device), y.to(self.device)
        history = []
        model.train()
        for epoch in range(epochs):
            #梯度清零
            optimizer.zero_grad()
            #计算损失
            loss = criterion(model(x), y)
            #反向传播
            loss.backward()
            #更新参数
            optimizer.step()
            history.append(loss.item())
            if epoch % 20 == 0 or epoch == epochs-1:
                print(f"epoch: {epoch}, loss: {loss}")
        self.model = model
        return history
    
    def predict(self, x):
        self.model.eval()
        x = x.to(self.device)
        with torch.no_grad():
            out = self.model(x).cpu()
        return out
    
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D = 8, 16, 64
    x = torch.randn(B, T, D)
    y = torch.sin(x.cumsum(dim=1))                  # 任意确定性目标，能过拟合即可
    transformer = Transformer()
    hist = transformer.train(x, y, epochs=200)
    print(f"loss: {hist[0]:.4f} -> {hist[-1]:.4f}")



# # 标准写法
# """T5 参考实现：Pre-LN decoder 块堆叠 + 手写训练循环（玩具回归任务）

# 对照你三稿的差异点（标 ★ 的地方）：
#   ★1  import torch.optim（不是 torch.optimizer）
#   ★2  Block / Model / Trainer 三个类平级定义，不嵌套
#       （嵌套也合法，但 Model 里裸写 Block 会 NameError，须写 Transformer.Block）
#   ★3  Model.forward 用 self.blocks（和 __init__ 里的名字一致），直接迭代
#   ★4  Trainer.train 里实例化 self.Model（三参签名），不是 self.Block
# """
# import torch
# import torch.nn as nn
# import torch.optim as optim

# from my_attention import MultiHeadAttention


# class Block(nn.Module):
#     """一个 Pre-LN decoder 块：attention 子层 + FFN 子层，各带独立 Norm 和残差"""

#     def __init__(self, d_model, n_heads):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(d_model)            # 子层①支路 Norm
#         self.attn = MultiHeadAttention(d_model, n_heads)
#         self.ln2 = nn.LayerNorm(d_model)            # 子层②支路 Norm，与①参数独立
#         self.fc1 = nn.Linear(d_model, 4 * d_model)
#         self.re = nn.ReLU()
#         self.fc2 = nn.Linear(4 * d_model, d_model)

#     def forward(self, x):
#         x = x + self.attn(self.ln1(x))              # 残差拿 LN 之前的 x（看结构图绿线分叉点）
#         x = x + self.fc2(self.re(self.fc1(self.ln2(x))))
#         return x


# class Model(nn.Module):
#     """整个模型：n_layers 个 Block 堆叠 + final Norm"""

#     def __init__(self, d_model, n_heads, n_layers):
#         super().__init__()
#         self.blocks = nn.ModuleList(
#             [Block(d_model, n_heads) for _ in range(n_layers)]
#         )                                           # ★ 普通 list 不注册参数，必须 ModuleList
#         self.norm_f = nn.LayerNorm(d_model)         # 全模型唯一的 final Norm，不属于任何 Block

#     def forward(self, x):
#         for block in self.blocks:
#             x = block(x)
#         return self.norm_f(x)


# class Trainer:
#     """训练器：持有 device 与超参，train 收敛、predict 推理"""

#     def __init__(self, n_heads=4, n_layers=2, lr=1e-3):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.n_heads = n_heads
#         self.n_layers = n_layers
#         self.lr = lr

#     def train(self, x, y, epochs=200, log_every=40):
#         model = Model(x.shape[-1], self.n_heads, self.n_layers).to(self.device)
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(model.parameters(), lr=self.lr)
#         x, y = x.to(self.device), y.to(self.device)
#         model.train()
#         history = []
#         for epoch in range(epochs):
#             optimizer.zero_grad()                   # ① 清梯度
#             loss = criterion(model(x), y)           # ② 前向 + 损失
#             loss.backward()                         # ③ 反向
#             optimizer.step()                        # ④ 更新
#             history.append(loss.item())
#             if epoch % log_every == 0 or epoch == epochs - 1:
#                 print(f"epoch {epoch:3d} | loss {loss.item():.6f}")
#         self.model = model
#         return history

#     @torch.no_grad()
#     def predict(self, x):
#         self.model.eval()
#         return self.model(x.to(self.device)).cpu()


# if __name__ == "__main__":
#     torch.manual_seed(0)
#     B, T, D = 8, 16, 64
#     x = torch.randn(B, T, D)
#     y = torch.sin(x.cumsum(dim=1))                  # 任意确定性目标，能过拟合即可
#     trainer = Trainer()
#     hist = trainer.train(x, y, epochs=200)
#     print(f"loss: {hist[0]:.4f} -> {hist[-1]:.4f}")

        