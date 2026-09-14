# -*- coding: utf-8 -*-
"""IMDB 脚手架：数据管线已给全（新知识，看懂），模型与循环留 TODO（你写过一遍的东西）

用法：python imdb_scaffold.py 先确认数据管线自检通过，再逐个填 TODO。
调试铁律：先用 500 条小样本把全流程跑通，再放开全量——别一上来 22000 条等半小时报个错。
"""

# 形状自检
# x          (B, 256)   int64   ← DataLoader 吐出来
# embed(x)   (B, 256, 128)
# 每个 Block 输出            (B, 256, 128)
# mean(dim=1)               (B, 128)
# fc                        (B, 2)      ← logits
# argmax(dim=-1)            (B,)        ← 预测



import re
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset   # 跑之前: export HF_ENDPOINT=https://hf-mirror.com

from train import Block

SEQ_LEN = 256
VOCAB_SIZE = 20000
PAD, UNK = 0, 1


# ---------- 第 1 步：加载与切分（已写好，看懂即可） ----------
def load_raw():
    """IMDB 原始数据长这样: {'text': 'This movie was...', 'label': 1}"""
    ds = load_dataset("stanfordnlp/imdb")
    train = ds["train"].shuffle(seed=42)
    n_val = 3000
    val = train.select(range(n_val))                 # 3000 条做验证（早停用）
    tr = train.select(range(n_val, len(train)))      # 其余 22000 条训练
    test = ds["test"]                                # 最终汇报才碰
    return tr, val, test


# ---------- 第 2 步：分词与词表（新知识，给全，读懂每行） ----------
_tok = re.compile(r"[a-z']+|[!?.,]")                 # 极简分词：小写单词或常见标点

def build_vocab(texts, max_size=VOCAB_SIZE):
    """统计词频 → 词表布局: 0=PAD, 1=UNK, 2 起是真实词（按词频降序）"""
    counter = Counter()
    for t in texts:
        counter.update(_tok.findall(t.lower()))
    itos = ["<pad>", "<unk>"] + [w for w, _ in counter.most_common(max_size - 2)]
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi

def encode(text, stoi):
    """字符串 → 固定长度索引序列：查词表(查不到=UNK)，截断，尾部补 PAD"""
    ids = [stoi.get(w, UNK) for w in _tok.findall(text.lower())]
    ids = ids[:SEQ_LEN]
    ids += [PAD] * (SEQ_LEN - len(ids))
    return ids


# ---------- 第 3 步：转张量（已写好） ----------
def to_tensors(split, stoi):
    xs = torch.tensor([encode(t, stoi) for t in split["text"]], dtype=torch.long)
    ys = torch.tensor(split["label"], dtype=torch.long)
    return xs, ys


# ---------- 第 4 步：模型（TODO 在这里，对照形状契约填） ----------
class Model(nn.Module):
    def __init__(self, vocab, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x):
        # TODO-1: embedding → 期望 (B, 256, 128)
        x = self.embed(x)
        # TODO-2: 逐个过 block。分类不需要因果掩码：
        #         提示——给 Block.forward 加透传参数 causal，一路传给你 MHA 的 forward；
        #         注意 MHA.forward 本来就收 causal，Block 只是转发。
        for block in self.blocks:
            x = block(x, False)
        # TODO-3: 时间维 mean pooling → (B, 128)   提示: x.mean(dim=?)  想清楚 dim 是哪个
        x = x.mean(dim=1) # 这里对dim=1维度做平均是因为截取的256个token要做平均打分
        # TODO-4: self.fc → (B, 2)
        return self.fc(x)


# ---------- 第 5 步：验证与训练循环（TODO） ----------
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            # TODO-5: 预测 = model(xb).argmax(dim=-1)，累计 correct 与 total
            output = model(xb).argmax(dim=-1) # 这里argmax(dim=-1)是对(B, 2)里的最后一个维度求出最大得分的id
            total += output.shape[0]
            correct += (output == yb).sum().item()
    return correct / total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr, val, test = load_raw() # 获取训练集22000条， 验证集3000条， 测试集25000条

    # # 调试期：小样本建词表 + 小样本训练，全流程通了再放开
    # stoi = build_vocab(tr["text"][:2000]) #用训练集前2000条构建词表
    # xs, ys = to_tensors(tr.select(range(500)), stoi)
    # vx, vy = to_tensors(val.select(range(500)), stoi)
    # print("形状自检:", xs.shape, xs.dtype, "|", ys.shape)   # 期望 (500,256) int64 | (500,)

    # # TODO-6: TensorDataset(xs, ys) + DataLoader(..., batch_size=32, shuffle=True)
    # #         验证集另一个 DataLoader，shuffle=False
    # tr_loader = DataLoader(TensorDataset(xs, ys), batch_size=32, shuffle=True)
    # val_loader = DataLoader(TensorDataset(vx, vy), batch_size=32, shuffle=False)
    # # TODO-7: model = Model(len(stoi)).to(device)；Adam(lr=3e-4)；CrossEntropyLoss()
    # model = Model(len(stoi)).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=3e-4)
    # criterion = nn.CrossEntropyLoss() 
    # val_acc_list = []
    # # TODO-8: 手写训练循环（T5 你写过）：每 batch zero_grad → forward → loss → backward → step
    # for epoch in range(4):
    #     epoch_loss = 0
    #     for tr_x, tr_y in tr_loader:
    #         tr_x, tr_y = tr_x.to(device), tr_y.to(device)
    #         optimizer.zero_grad()
    #         loss = criterion(model(tr_x, False), tr_y)
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.item()
    #     # TODO-9: 每 epoch 末 evaluate 打印 val acc；创新高就 torch.save(model.state_dict(), "best_model.pth")
    #     val_acc = evaluate(model, val_loader, device)
    #     print(f"epoch: {epoch}, epoch_train_loss: {epoch_loss / len(tr_loader)}, val_acc: {val_acc}")
    #     if not val_acc_list or val_acc > max(val_acc_list):
    #         torch.save(model.state_dict(), "best_model.pth")
    #     val_acc_list.append(val_acc)
    # 小样本通关后放开全量：
    #   stoi = build_vocab(tr["text"])；to_tensors(tr/val)；重训 3-4 epoch
    #   最后 model.load_state_dict(torch.load("best_model.pth")) + evaluate(test)
    
    
    stoi = build_vocab(tr["text"]) #用全部训练集构建词表
    xs, ys = to_tensors(tr, stoi)
    vx, vy = to_tensors(val, stoi)
    tx, ty = to_tensors(test, stoi)
    print("形状自检:", xs.shape, xs.dtype, "|", ys.shape)   # 期望 (22000,256) int64 | (22000,)

    # TODO-6: TensorDataset(xs, ys) + DataLoader(..., batch_size=32, shuffle=True)
    #         验证集另一个 DataLoader，shuffle=False
    tr_loader = DataLoader(TensorDataset(xs, ys), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(vx, vy), batch_size=32, shuffle=False)
    # TODO-7: model = Model(len(stoi)).to(device)；Adam(lr=3e-4)；CrossEntropyLoss()
    model = Model(len(stoi)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss() 
    val_acc_list = []
    # TODO-8: 手写训练循环（T5 你写过）：每 batch zero_grad → forward → loss → backward → step
    for epoch in range(4):
        epoch_loss = 0
        for tr_x, tr_y in tr_loader:
            tr_x, tr_y = tr_x.to(device), tr_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(tr_x), tr_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # TODO-9: 每 epoch 末 evaluate 打印 val acc；创新高就 torch.save(model.state_dict(), "best_model.pth")
        val_acc = evaluate(model, val_loader, device)
        print(f"epoch: {epoch}, epoch_train_loss: {epoch_loss / len(tr_loader)}, val_acc: {val_acc}")
        if not val_acc_list or val_acc > max(val_acc_list):
            torch.save(model.state_dict(), "best_model.pth")
        val_acc_list.append(val_acc)
    
    model.load_state_dict(torch.load("best_model.pth"))
    test_loader = DataLoader(TensorDataset(tx, ty), batch_size=32, shuffle=False)
    test_acc = evaluate(model, test_loader, device)
    print(f'test_acc: {test_acc}')


if __name__ == "__main__":
    main()

