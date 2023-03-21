# 网络流量分析课程示例 - 基于原型网络的小样本加密流量分类

# From Scratch

1. 下载 [VPN-nonVPN (ISCXVPN2016)](https://www.unb.ca/cic/datasets/vpn.htm) 数据集
2. 处理数据得到流量载荷：[DeepTraffic/encrypted_traffic_classification/](https://github.com/echowei/DeepTraffic/tree/master/2.encrypted_traffic_classification)
3. 将得到的载荷数据转化为日志文件：
    ```
    // 每行一条 json 日志
    ...
    {"pcap_name": "AIMchat1", "vpn_label": "non-vpn", "service_label": "chat", "data": [161, 178, ..., 124, 239]}
    ...
    ```

**或者使用已经处理好的数据 `iscx2016_vpn_payloadl7_784`**：

1. 解压 iscx2016_vpn.zip : `unzip iscx2016_vpn.zip`
2. 拷贝 `iscx2016_vpn_payloadl7_784.log` 文件到项目根目录


# Usage

运行代码：`python protonet.py`

> 参数说明：
> - path: 数据存放路径，默认为 `./iscx2016_vpn_payloadl7_784.log`
> - max_epoch: 最大训练迭代次数
> - train_way, shot, train_query: 训练使用的 n-way k-shot 配置，以及查询集样本个数
> - test_way, test_shot, test_query: 测试使用的  n-way k-shot 配置，以及测试样本个数

&nbsp;

附：执行结果

```
Namespace(gpu=0, max_epoch=5, path='iscx2016_vpn_payloadl7_784.log', test_query=30, test_shot=10, test_way=2, train_query=15, train_shot=10, train_way=2)
ISCX2016 dataset X.shape: torch.Size([54053, 1, 784]), Y.shape: torch.Size([54053])
Settings: train classes: ['vpn_filetransfer', 'vpn_streaming', 'filetransfer', 'email']
Settings: valid classes: ['vpn_chat', 'streaming', 'voip', 'vpn_email']
Settings: test classes: ['vpn_voip', 'p2p', 'vpn_p2p', 'chat']
epoch 1: train_acc: 0.9357      val_acc: 0.9183
epoch 2: train_acc: 0.9603      val_acc: 0.9133
epoch 3: train_acc: 0.9467      val_acc: 0.9300
epoch 4: train_acc: 0.9743      val_acc: 0.9467
epoch 5: train_acc: 0.9680      val_acc: 0.9117
batch 1 test_acc: 86.67
batch 2 test_acc: 90.83
batch 3 test_acc: 93.89
batch 4 test_acc: 95.42
batch 5 test_acc: 94.00
batch 6 test_acc: 95.00
batch 7 test_acc: 95.00
batch 8 test_acc: 95.00
batch 9 test_acc: 92.41
batch 10 test_acc: 92.83
```