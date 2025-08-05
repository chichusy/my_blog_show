---
date : '2025-08-05T14:44:25+08:00'
draft : false
title : '线性回归解析解的推导'
image: index.jpg
categories:
  - 学习
tags:
  - 工程实践
---
# 创建一个深度学习虚拟环境（包含d2l包）

---

## Step1

创建一个名为“d2l_env”的虚拟环境并激活（推荐 Python 3.9，兼容性最佳）：

```python
conda create -n d2l_env python=3.9
```

```python
conda activate d2l_env
```

## Step2

使用pip安装pytorch（最新版的pytorch已经不支持conda安装，故采用pip）：

```python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Step3

使用conda安装常见相关依赖

```python
conda install matplotlib pandas jupyter ipykernel
```

## Step4

使用pip安装指定版本的d2l包：

```python
pip install d2l==1.0.2
```

## Step5

将这个虚拟环境加入jupyter notebook内核：

```python
python -m ipykernel install --user --name d2l_env --display-name "Python (d2l_env)"
```

