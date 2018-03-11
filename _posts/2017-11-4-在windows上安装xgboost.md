---
layout: post
title: "在windows上安装xgboost"
date: 2017-11-4 10:00:00 +0800
categories: python
tags: 库安装
description: windows上安装xgboost的步骤
---

# 在windows上安装xgboost

## 1. github上下载xgboost的编译文件

```powershell
cd yourpath
git clone --recursive https://github.com/dmlc/xgboost
```

```powershell
cd yourpath/xgboost
mkdir build
cd build
cmake .. -G "Visual Studio 15 2017 Win64"
```

visual studio 打开build文件夹下生成的xgboost.sln文件，之后运行

生成->清理解决方案

生成->生成解决方案

生成完毕后，将yourpath/xgboost/lib下的xgboost.dll文件复制到yourpath/xgboost/python-package文件夹下。

之后运行

```Powershell
cd yourpath/xgboost/python-package
python setup.py install
```

