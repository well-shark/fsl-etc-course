import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import iscx2016


def pairwise_distances_logits(a, b):
    '''Computes pairwise distances between a and b.
    '''
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits


def accuracy(predictions, targets):
    '''Computes classification accuracy.
    '''
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


class CNN1D(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(CNN1D, self).__init__()
        # 定义神经网络结构
        self.C1 = nn.Conv1d(in_channels=in_channels, out_channels=32,
                            kernel_size=2, stride=1, padding=1, bias=True)
        self.R1 = nn.ReLU()
        self.B1 = nn.BatchNorm1d(32)
        self.S2 = nn.MaxPool1d(kernel_size=2)
        self.C2 = nn.Conv1d(32, 64, 2, 1, 1, bias=True)
        self.R2 = nn.ReLU()
        self.B2 = nn.BatchNorm1d(64)
        self.S4 = nn.MaxPool1d(2)
        self.F1 = nn.Flatten()
        self.F2 = nn.Linear(in_features=12544, out_features=1024)
        self.D1 = nn.Dropout(p=0.1)
        self.OUT = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.C1(x)
        x = self.R1(x)
        x = self.B1(x)
        x = self.S2(x)
        x = self.C2(x)
        x = self.R2(x)
        x = self.B2(x)
        x = self.S4(x)
        x = self.F1(x)
        x = self.F2(x)
        x = self.D1(x)
        x = self.OUT(x)
        return x


def fast_adapt(model, batch, ways, shot, query_num, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)
    n_items = shot * ways

    # Sort data samples by labels
    # TODO: Can this be replaced by ConsecutiveLabels ?
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    embeddings = model(data)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = embeddings[query_indices]
    labels = labels[query_indices].long()

    logits = pairwise_distances_logits(query, support)
    loss = F.cross_entropy(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='iscx2016_vpn_payloadl7_784.log')
    parser.add_argument('--max-epoch', type=int, default=5)
    parser.add_argument('--train-way', type=int, default=2)
    parser.add_argument('--train-shot', type=int, default=10)
    parser.add_argument('--train-query', type=int, default=15)
    parser.add_argument('--test-way', type=int, default=2)
    parser.add_argument('--test-shot', type=int, default=10)
    parser.add_argument('--test-query', type=int, default=30)
    parser.add_argument('--gpu', default=0)
    args = parser.parse_args()
    print(args)

    device = torch.device('cpu')
    if args.gpu and torch.cuda.device_count():
        print("Using gpu")
        torch.cuda.manual_seed(43)
        device = torch.device('cuda')

    model = CNN1D(1, 12)
    model.to(device)

    # 加载数据集
    path_data = args.path
    tasksets = iscx2016.get_taskset(path_data, 
        train_ways=args.train_way, train_samples=args.train_shot+args.train_query, 
        test_ways=args.test_way, test_samples=args.test_shot+args.test_query,
        num_tasks=(-1, 10, 10), device=None,
        input_channels=1, input_length=784)

    train_loader = DataLoader(tasksets.train, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(tasksets.valid, pin_memory=True, shuffle=True)
    test_loader = DataLoader(tasksets.test, pin_memory=True, shuffle=True)
    
    # 定义优化方法
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5)

    # 训练
    for epoch in range(1, args.max_epoch + 1):
        model.train()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0

        for i in range(100):
            batch = next(iter(train_loader))

            loss, acc = fast_adapt(model,
                                   batch,
                                   args.train_way,
                                   args.train_shot,
                                   args.train_query,
                                   metric=pairwise_distances_logits,
                                   device=device)

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        print(f'epoch {epoch}: train_acc: {n_acc/loss_ctr:.4f}', end='\t')

        model.eval()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        for i, batch in enumerate(valid_loader):
            loss, acc = fast_adapt(model,
                                   batch,
                                   args.test_way,
                                   args.test_shot,
                                   args.test_query,
                                   metric=pairwise_distances_logits,
                                   device=device)

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc
        print(f'val_acc: {n_acc/loss_ctr:.4f}')

    # 测试
    loss_ctr = 0
    n_acc = 0

    for i, batch in enumerate(test_loader, 1):
        loss, acc = fast_adapt(model,
                               batch,
                               args.test_way,
                               args.test_shot,
                               args.test_query,
                               metric=pairwise_distances_logits,
                               device=device)
        loss_ctr += 1
        n_acc += acc
        print('batch {} test_acc: {:.2f}'.format(i, n_acc/loss_ctr * 100))