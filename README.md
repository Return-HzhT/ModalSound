# 神经网络音效系统实验文档

## 数据集处理

依次调用dataset_scripts内以下程序进行数据处理

1. voxelize.py：mesh(obj) -> voxel
2. voxel_check.py：voxel ->_connect.npy
3. modal_analysis.py: voxel -> eigen, ffat, residual
4. residual_check.py: residual -> residual_ok.npy
5. voxel2pcd.py：voxel, eigen, ffat -> pcd_dataset，normalized_pcd_dataset（采样点云，特征值、特征向量和ffat的数据规范化）
6. eigen_mask.py：scaled_pcd_dataset -> final_pcd_dataset（特征值掩码）
7. splitDataset.py：pcd, pcd_eigen, ffat_map->train, test (划分训练集、验证集和测试集) 

## 神经网络

特征值、特征向量、ffat的三个神经网络相关代码分别位于eigenvalue_model、eigenvector_model、ffat_model文件夹下。

- dataset.py：自定义pyG数据集
- train.py：训练模型
- eval.py：评估模型

关于神经网络的定义和训练，尝试过的网络架构：pointnet, dgcnn, pointnet++, randlanet, point transformer，根据最终的效果选择了现在的网络架构

尝试了旋转的数据增广，从结果上看并不利于网络的学习；将特诊值对应的频率的范围缩小到50-5000Hz，从demo上看声音效果没有明显的改善

## 音效系统demo

generate.py：.obj -> .npz（网络预测的特征值、特征向量、ffat）

音效系统demo代码位于网盘：链接：https://pan.baidu.com/s/1j7bxrKbRuGhSz4xgvNGq4A 提取码：yywm 

其中unityProject为Unity前端，send为python后端，需要在Unity中指定物体的objectID，在后端准备好生成的数据即可运行。

参考的Unity VR教程：[Unity VR 开发教程 OpenXR+XR Interaction Toolkit (一) 安装和配置-CSDN博客](https://blog.csdn.net/qq_46044366/article/details/126676551)