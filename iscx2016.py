import json
import random
from typing import Union
from collections import namedtuple

import torch
import learn2learn as l2l
from sklearn import preprocessing
from torch.utils.data import Dataset

BenchmarkTasksets = namedtuple('BenchmarkTasksets', ('train', 'valid', 'test'))

LABEL_LIST = ['chat', 'email', 'filetransfer', 'p2p', 'streaming', 'voip', 
              'vpn_chat', 'vpn_email', 'vpn_filetransfer', 'vpn_p2p', 'vpn_streaming', 'vpn_voip']

class ISCX2016(Dataset):

    def __init__(self, raw_file:str, input_channels:int=1, input_length:int=784):
        self.raw_file = raw_file
        self.input_channels = input_channels
        self.input_length = input_length
        self.load_labels()

        self.X, self.y = self.load_data()
        self.len = len(self.y)
        print(f'ISCX2016 dataset X.shape: {self.X.shape}, Y.shape: {self.y.shape}')

    def load_labels(self):
        self.label_dict = {k:i for i, k in enumerate(LABEL_LIST)}
        self.labels = list(range(0, 12))
        self.n_classes = 12
    
    def load_data(self):
        X, y = [], []
        with open(self.raw_file, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                flow = json.loads(line)
                X.append(flow['data'])
                y.append(self.label_dict[flow['service_label']])

        X = preprocessing.scale(X)
        X = torch.tensor(X, dtype=torch.float).reshape(-1, 1, self.input_length)
        y = torch.tensor(y)
        return X, y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


def get_taskset(raw_file:str, train_ways:int, train_samples:int, 
    test_ways:int, test_samples:int, num_tasks:Union[int, tuple]=-1, device:str=None, **kwargs) -> BenchmarkTasksets:
    """Returns the taskset for a particular benchmark, using literature standard data and task transformations.

    The returned object is a namedtuple with attributes `train`, `valid`, `test` which
    correspond to their respective TaskDatasets.

    Args:
        root (str): Where the data is stored.
        train_ways (int): The number of classes per train tasks.
        train_samples (int): The number of samples per train classes.
        test_ways (int): The number of classes per test or validation tasks.
        test_samples (int): The number of classes per test or validation classes.
        num_tasks (int, optional): The number of tasks in each TaskDataset. Defaults to -1.
        device (str, optional): If not None, tasksets are loaded as Tensors on `device`. Defaults to None.
        **kwargs: Additional arguments passed to the TaskDataset.

    Example:

    ```python
    train_tasks, validation_tasks, test_tasks = get_taskset()
    batch = train_tasks.sample(**kwargs)
    or:
    tasksets = get_taskset(**kwargs)
    batch = tasksets.train.sample()
    ```

    """
    if device is not None:
        raise NotImplementedError('Device other than None not implemented. (yet)')

    if type(num_tasks) is int:
        num_train_tasks, num_valid_tasks, num_test_tasks = num_tasks, num_tasks, num_tasks
    elif type(num_tasks) is tuple:
        num_train_tasks, num_valid_tasks, num_test_tasks = num_tasks
    
    input_channels = kwargs.get('input_channels', 1)
    input_length = kwargs.get('input_length', 784)
    iscx2016 = ISCX2016(raw_file, input_channels=input_channels, input_length=input_length)
    dataset = l2l.data.MetaDataset(iscx2016)
    
    classes = iscx2016.labels
    random.shuffle(classes)
    train_labels, valid_labels, test_labels = classes[:4], classes[4:8], classes[8:]
    print(f"Settings: train classes: {[LABEL_LIST[i] for i in train_labels]}")
    print(f"Settings: valid classes: {[LABEL_LIST[i] for i in valid_labels]}")
    print(f"Settings: test classes: {[LABEL_LIST[i] for i in test_labels]}")
    train_dataset = l2l.data.FilteredMetaDataset(dataset, labels=train_labels)
    valid_dataset = l2l.data.FilteredMetaDataset(dataset, labels=valid_labels)
    test_dataset = l2l.data.FilteredMetaDataset(dataset, labels=test_labels)

    train_transforms = [
        l2l.data.transforms.FusedNWaysKShots(train_dataset, n=train_ways, k=train_samples),
        l2l.data.transforms.LoadData(train_dataset),
        l2l.data.transforms.RemapLabels(train_dataset),
        l2l.data.transforms.ConsecutiveLabels(train_dataset),
    ]

    valid_transforms = [
        l2l.data.transforms.FusedNWaysKShots(valid_dataset, n=test_ways, k=test_samples),
        l2l.data.transforms.LoadData(valid_dataset),
        l2l.data.transforms.RemapLabels(valid_dataset),
        l2l.data.transforms.ConsecutiveLabels(valid_dataset),
    ]

    test_transforms = [
        l2l.data.transforms.FusedNWaysKShots(test_dataset, n=test_ways, k=test_samples),
        l2l.data.transforms.LoadData(test_dataset),
        l2l.data.transforms.RemapLabels(test_dataset),
        l2l.data.transforms.ConsecutiveLabels(test_dataset),
    ]

    train_tasks = l2l.data.TaskDataset(
        dataset=train_dataset,
        task_transforms=train_transforms,
        num_tasks=num_train_tasks,
    )

    valid_tasks = l2l.data.TaskDataset(
        dataset=valid_dataset,
        task_transforms=valid_transforms,
        num_tasks=num_valid_tasks,
    )

    test_tasks = l2l.data.TaskDataset(
        dataset=test_dataset,
        task_transforms=test_transforms,
        num_tasks=num_test_tasks,
    )

    return BenchmarkTasksets(train_tasks, valid_tasks, test_tasks)