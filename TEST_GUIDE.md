# GatedDeltaNet 本地测试指南

## 环境安装

```bash
# 1. 克隆仓库
git clone https://github.com/hhaAndroid/gated_deltanet.git
cd gated_deltanet

# 2. 创建 conda 环境
conda create -n gated_deltanet python=3.12
conda activate gated_deltanet

# 3. 安装 PyTorch (用你的代理加速)
pip install torch==2.8.0

# 4. 安装依赖库
pip install causal-conv1d
pip install flash-linear-attention

# 5. 安装本项目
pip install -e .
```

## 快速测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行单个测试
python -m pytest tests/test_model.py::TestGatedDeltaNet -v
python -m pytest tests/test_model.py::TestGatedDeltaNetVarlen -v
python -m pytest tests/test_model.py::TestConsistency -v
```

## 手动验证

```python
import torch
from gated_deltanet import GatedDeltaNetConfig, GatedDeltaNet, GatedDeltaNetVarlen

# 测试基础版本
config = GatedDeltaNetConfig(hidden_size=64, num_key_heads=2, num_value_heads=2)
model = GatedDeltaNet(config)
x = torch.randn(2, 16, 64)  # (batch, seq_len, hidden)
out = model(x)
print(f"Base output shape: {out.shape}")  # (2, 16, 64)

# 测试 varlen 版本
model_varlen = GatedDeltaNetVarlen(config)
# 3 个序列，长度分别为 5, 10, 7
cu_seqlens = torch.tensor([0, 5, 15, 22], dtype=torch.int32)
x_varlen = torch.randn(22, 64)  # (total_tokens, hidden)
out_varlen = model_varlen(x_varlen, cu_seqlens)
print(f"Varlen output shape: {out_varlen.shape}")  # (22, 64)

# 验证一致性
# 把等长序列分别输入两个版本，结果应该相同
x_batch = torch.randn(2, 16, 64)
cu_seqlens_equal = torch.tensor([0, 16, 32], dtype=torch.int32)
x_packed = x_batch.reshape(32, 64)

out_base = model(x_batch)
out_varlen_packed = model_varlen(x_packed, cu_seqlens_equal)
out_varlen_batch = out_varlen_packed.reshape(2, 16, 64)

print(f"Max diff: {(out_base - out_varlen_batch).abs().max().item()}")
# 应该在 1e-4 或更小的范围内
```

## 预期结果

如果实现正确：
- 基础版本输出: `(batch, seq_len, hidden_size)`
- Varlen 版本输出: `(total_tokens, hidden_size)`
- 两版本在相同输入下输出差异 < 1e-4

## 常见问题

### 1. causal_conv1d 安装失败
```bash
# 如果没有 CUDA，使用 CPU 版本
pip install causal-conv1d --no-build-isolation
```

### 2. flash-linear-attention 安装失败
```bash
# 从源码安装
git clone https://github.com/fla-org/flash-linear-attention.git
cd flash-linear-attention
pip install -e .
```

### 3. 测试失败
检查：
- PyTorch 版本 >= 2.0
- CUDA 版本与 PyTorch 匹配（如果使用 GPU）
- 所有依赖已正确安装

## 代码结构

```
gated_deltanet/
├── config.py         # 配置类 (32行)
├── model.py          # 基础训练版本 (305行)
├── model_varlen.py   # 变长训练版本 (415行)
└── __init__.py       # 包初始化 (18行)
```

**总计: 770 行**
