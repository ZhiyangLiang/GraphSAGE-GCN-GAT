## Introduction

这个仓库是我对在参考网络上的代码后，对GraphSAGE、GCN、GAT的代码实现/整合。

junior模块：这是我最初写的一些比较零散的代码块，主要用于测试。

GraphSAGE、GCN、GAT：分别是三个神经网络的代码实现，每个文件夹都具有数据导入、网络架构和训练测试三个模块。

final模块：我以GCN和GAT处理数据集的方式，分别提供了三种网络的训练接口，当然还有训练结果的绘图操作（经测试，在该处理方式下，GraphSAGE的性能有所下降）。

GraphSAGE_final模块：我以GraphSAGE为主干网络，GCN和GAT的训练思想融入到GraphSAGE的采样方式中，即代码中的GCNAggregator和GATAggregator。除此以外，还有GraphSAGE常用的：MeanAggregator、MaxAggregator、SumAggregator可供选择。

GraphSAGE主干网络在训练过程中使用了两个采样器，你们可以这五种采样器中自行选择搭配（经测试，只要两个采样器不同时采用SumAggregator，最终的test_acc一般能达到0.85及以上），当然你们也可以尝试去增加或减少使用采样器的数目。

更多细节可见于我在知乎写的一篇文章：https://zhuanlan.zhihu.com/p/560962482。

## Usage

将该仓库的文件克隆到本地：

```
git clone git@github.com:paradox-11/GNN.git
```

下载所需的框架与第三方库：

```
pip install torch==1.12.1
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.12.1+cpu.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.12.1+cpu.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.12.1+cpu.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.12.1+cpu.html
pip install torch-geometric
```

对于任何代码实现中不理解的细节，欢迎Issue~
