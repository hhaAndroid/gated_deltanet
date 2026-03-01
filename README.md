# GatedDeltaNet

基于 Qwen3.5 MoE 的 GatedDeltaNet 实现，支持变长序列（varlen）。

## 功能特性

- **GatedDeltaNet**: 基础版本，支持标准 batch 输入
- **GatedDeltaNetVarlen**: 变长序列版本，支持不同长度的序列在一个 batch 中高效处理
- 完整的单元测试，验证两种实现的数值一致性

## 环境要求

- Python 3.12+
- PyTorch 2.8+
- CUDA (适配你的显卡驱动)
- causal-conv1d
- flash-linear-attention

## 安装

```bash
# 创建 conda 环境
conda create -n gated_deltanet python=3.12
conda activate gated_deltanet

# 安装 PyTorch (根据你的 CUDA 版本调整)
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu121

# 安装依赖
pip install causal-conv1d
pip install flash-linear-attention

# 安装本项目
pip install -e .
```

## 使用示例

```python
import torch
from gated_deltanet import GatedDeltaNetConfig, GatedDeltaNet, GatedDeltaNetVarlen

# 创建配置
config = GatedDeltaNetConfig(
    hidden_size=512,
    num_key_heads=4,
    num_value_heads=4,
    key_head_dim=128,
    value_head_dim=128,
    conv_kernel_size=4,
)

# 基础版本
model = GatedDeltaNet(config, layer_idx=0).cuda()
x = torch.randn(2, 100, 512).cuda()  # (batch, seq_len, hidden)
out = model(x)

# 变长版本
model_varlen = GatedDeltaNetVarlen(config, layer_idx=0).cuda()
# cu_seqlens 表示每个序列的累积长度
cu_seqlens = torch.tensor([0, 50, 100], dtype=torch.int32).cuda()
out_varlen = model_varlen(x, cu_seqlens=cu_seqlens)
```

## 测试

```bash
pytest tests/ -v
```

## 项目结构

```
gated_deltanet/
├── gated_deltanet/
│   ├── __init__.py
│   ├── config.py         # 配置类
│   ├── model.py          # GatedDeltaNet 实现
│   └── model_varlen.py   # 变长版本实现
├── tests/
│   └── test_model.py     # 单元测试
├── setup.py
└── README.md
```

## 参考

- [transformers Qwen3.5 MoE](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
