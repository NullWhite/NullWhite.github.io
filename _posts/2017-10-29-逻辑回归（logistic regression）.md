---
layout: post
title: "逻辑回归(logistic regression)"
date: 2017-10-29 20:40:00 +0800
categories: 机器学习
tags: 分类方法
description: 逻辑回归的原理以及向量化表示
---

# 逻辑回归（logistic regression）

## 使用情形

用于分类（classfication）问题求解。

## 基本概念

### 模型（hypothesis）

$$
h(z)=\frac{1}{1+e^{-z}}
$$

$$
z=\theta x+\omega
$$

式中：

$$\theta$$，$$\omega$$是模型的参数；

$$x$$是模型的输入。

### 代价函数（cost function）

$$
cost(x)=-y\log\hat y-(1-y)\log(1-\hat y)
$$

### 损失函数（loss function）[无正则化]

$$
J(\theta,\omega)=-\frac{1}{m}\sum_{i=0}^m\left(y^{(i)}\log h\left(x^{(i)}\right)+\left(1-y\right)\log \left(1-h\left(x^{(i)}\right)\right)\right)
$$

式中:

$$m$$为训练样本的数量；

$$x^{(i)}$$表示第$$i$$组训练数据；

$$y^{(i)}$$表示第$$i$$组训练数据的label值。

### 损失函数（loss function）[正则化]

$$
J(\theta,\omega)=-\frac{1}{m}\sum_{i=1}^m\left(y^{(i)}\log h\left(x^{(i)}\right)+(1-y)\log \left(1-h\left(x^{(i)}\right)\right)\right)+\frac{\lambda}{2m}\sum_{i=1}^n{\theta_i}^2
$$

式中：

$$\lambda$$是超参数；

$$n$$是参数$$\theta$$的数量（维数）；

$$\theta_i$$表示第i个$$\theta$$。


### 目标函数（object function）

$$
\mathop{minimize}\limits_{\theta,\omega}{J(\theta,\omega)}
$$

### 梯度（无正则化）

$$
\begin{align}\frac{d_{J(\theta,\omega)}}{d_{\theta_j}}&=\frac{d_{J(\theta,\omega)}}{d_{h(x)}}\frac{d_{h(x)}}{d_z}\frac{d_z}{d_{\theta_j}} \\
&=\frac{1}{m}\displaystyle\sum_{i=1}^m\left(\left(-\frac{y^{(i)}}{h\left(x^{(i)}\right)}+\frac{1-y}{1-h\left(x^{(i)}\right)}\right)\left(h\left(x^{(i)}\right)(1-h\left(x^{(i)}\right)\right)\left(x_j^{(i)}\right)\right)\\
&=\frac{1}{m}\displaystyle\sum_{i=1}^m\left(\left(h\left(x^{(i)}\right)-y^{(i)}\right)x_j^{(i)}\right)\end{align}
$$

$$
\begin{align}\frac{d_{J(\theta,\omega)}}{d_{\omega}}&=\frac{d_{J(\theta,\omega)}}{d_{h(x)}}\frac{d_{h(x)}}{d_z}\frac{d_z}{d_{\omega}} \\
&=\frac{1}{m}\displaystyle\sum_{i=1}^m\left(\left(-\frac{y^{(i)}}{h\left(x^{(i)}\right)}+\frac{1-y}{1-h\left(x^{(i)}\right)}\right)\left(h\left(x^{(i)}\right)(1-h\left(x^{(i)}\right)\right)\right)\\
&=\frac{1}{m}\displaystyle\sum_{i=1}^m\left(h\left(x^{(i)}\right)-y^{(i)}\right)\end{align}
$$

式中:

$$x_j^{(i)}$$表示第$$i$$组训练数据的第$$j$$个特征值。

### 梯度（正则化）

$$
\begin{align}\frac{d_{J(\theta,\omega)}}{d_{\theta_j}}&=\frac{d_{J(\theta,\omega)}}{d_{h(x)}}\frac{d_{h(x)}}{d_z}\frac{d_z}{d_{\theta_j}} \\
&=\frac{1}{m}\displaystyle\sum_{i=1}^m\left(\left(-\frac{y^{(i)}}{h\left(x^{(i)}\right)}+\frac{1-y}{1-h\left(x^{(i)}\right)}\right)\left(h\left(x^{(i)}\right)(1-h\left(x^{(i)}\right)\right)\left(x_j^{(i)}\right)\right)+\frac{\lambda}{m}\theta_j\\
&=\frac{1}{m}\displaystyle\sum_{i=1}^m\left(\left(h\left(x^{(i)}\right)-y^{(i)}\right)x_j^{(i)}\right)+\frac{\lambda}{m}\theta_j\end{align}
$$

$$
\begin{align}\frac{d_{J(\theta,\omega)}}{d_{\omega}}&=\frac{d_{J(\theta,\omega)}}{d_{h(x)}}\frac{d_{h(x)}}{d_z}\frac{d_z}{d_{\omega}} \\
&=\frac{1}{m}\displaystyle\sum_{i=1}^m\left(\left(-\frac{y^{(i)}}{h\left(x^{(i)}\right)}+\frac{1-y}{1-h\left(x^{(i)}\right)}\right)\left(h\left(x^{(i)}\right)(1-h\left(x^{(i)}\right)\right)\right)\\
&=\frac{1}{m}\displaystyle\sum_{i=1}^m\left(h\left(x^{(i)}\right)-y^{(i)}\right)\end{align}
$$

式中:

$$\lambda$$是正则化权重。

### 梯度下降

$$
\theta_j=\theta_j-\alpha\frac{d_{J(\theta,\omega)}}{d_{\theta_j}}
$$

$$
\omega=\omega-\alpha\frac{d_{J(\theta,\omega)}}{d_{\theta_j}}
$$

式中：

$$\alpha$$是学习速率。

## 向量化形式（vectorization）

$$
h(z)=\frac{1}{1+e^{-z}}
$$

$$
z=x\cdot\theta+\omega
$$

__无正则化：__

$$
J(\theta,\omega)=-\frac{1}{m}\left(y^T\cdot\log h(x)+(1-y)^T\cdot\log(1-h(x))\right)
$$

$$
d_\theta=\frac{1}{m}x^T\cdot\left(h(x)-y\right)
$$

$$
d_\omega=\frac{1}{m}\mathop{sum}\left(h(x)-y\right)
$$

$$
\theta=1-\alpha d_\theta
$$

$$
\omega=1-\alpha d_\omega
$$

__正则化：__

$$
J(\theta,\omega)=-\frac{1}{m}\left(y^T\cdot\log h(x)+(1-y)^T\cdot\log(1-h(x))\right)+\frac{\lambda}{2m}\theta^T\cdot\theta
$$

$$
d_\theta=\frac{1}{m}x^T\cdot\left(h(x)-y\right)+\frac{\lambda}{m}\theta
$$

$$
d_\omega=\frac{1}{m}\mathop{sum}\left(h(x)-y\right)
$$

$$
\theta=1-\alpha d_\theta
$$

$$
\omega=1-\alpha d_\omega
$$

式中：

$$m$$，$$n$$分别为训练集的大小和特征数；

$$x$$为$$m\times n$$维矩阵，每一行为一组特征数为$$n$$的数据，共有$$m$$组数据；

$$\theta$$为$$n$$维列向量；

$$\omega$$为浮点数；

$$z$$为$$n$$维列向量；

$$h(z)$$为$$n$$维列向量；

$$y$$为$$n$$维列向量；

$$J(\theta,\omega)$$为浮点数；

$$d_\theta$$为$$n$$维列向量；

$$\omega$$为浮点数；

$$c=a\cdot b$$表示$$a$$与$$b$$矩阵相乘，即$$c_{i,i}=a_{i,j}\times b_{j,i}$$

$$c=a\times b$$表示$$a$$与$$b$$元素对应相乘，即$$c_{i,j}=a_{i,j}\times b_{i,j}$$