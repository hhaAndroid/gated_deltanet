#!/bin/bash
# format.sh - 自动格式化代码

echo "🔧 格式化代码..."

# 检查是否在正确的目录
if [ ! -d "gated_deltanet" ]; then
    echo "❌ 错误: 请在仓库根目录运行此脚本"
    exit 1
fi

echo ""
echo "1️⃣ 使用 black 格式化代码..."
black gated_deltanet/ tests/ || {
    echo "⚠️ black 未安装，正在安装..."
    pip install black
    black gated_deltanet/ tests/
}

echo ""
echo "2️⃣ 使用 isort 排序导入..."
isort gated_deltanet/ tests/ || {
    echo "⚠️ isort 未安装，正在安装..."
    pip install isort
    isort gated_deltanet/ tests/
}

echo ""
echo "✅ 格式化完成！"
echo ""
echo "现在可以提交代码了:"
echo "  git add ."
echo "  git commit -m 'style: format code with black and isort'"
