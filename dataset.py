import os
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# Mini-ImageNet Dataset
class MiniImagenet(Dataset):
    """
    数据组织格式：
    miniimagenet/
      ├── images/                # 所有图像
      ├── train.csv              # CSV格式：filename,label
      ├── val.csv
      └── test.csv

    这里构建任务：每个任务包含 n_way 类别，每个类别有 k_shot 个支持样本与 k_query 个查询样本。
    """
    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize=84, startidx=0):
        self.batchsz = batchsz          # 一个 epoch 中任务数
        self.n_way = n_way              # n-way 分类
        self.k_shot = k_shot            # k-shot 支持集
        self.k_query = k_query          # 每类的查询样本数
        self.resize = resize            # 图像 resize 尺寸
        self.startidx = startidx        # 标签起始索引
        self.setsz = n_way * k_shot     # 每个任务支持集样本数
        self.querysz = n_way * k_query  # 每个任务查询集样本数

        print(f'Loading {mode} set: {batchsz} tasks, {n_way}-way, {k_shot}-shot, {k_query}-query, resize: {resize}')
        self.transform = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.img_path = os.path.join(root, 'images')
        csvfile = os.path.join(root, mode + '.csv')
        self.data, self.img2label = self.load_csv(csvfile)
        self.cls_num = len(self.data)

    def load_csv(self, csvf):
        data = {}
        img2label = {}
        with open(csvf, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # 跳过标题
            for row in reader:
                filename, label = row[0], row[1]
                if label not in data:
                    data[label] = []
                data[label].append(filename)
        # 建立类别到数值标签的映射
        labels = sorted(data.keys())
        for i, label in enumerate(labels):
            img2label[label] = i + self.startidx
        return list(data.values()), img2label

    def __getitem__(self, index):
        # 动态随机采样支持集与查询集，每次调用 __getitem__ 都生成一个全新的任务
        support_imgs = torch.zeros((self.setsz, 3, self.resize, self.resize))
        query_imgs = torch.zeros((self.querysz, 3, self.resize, self.resize))
        
        # 随机选择 n_way 个类别
        selected_cls = np.random.choice(self.cls_num, self.n_way, replace=False)
        support_batch = []
        query_batch = []
        for cls in selected_cls:
            # 每个类别中随机选择 k_shot + k_query 个样本
            indices = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, replace=False)
            support_batch.append(np.array(self.data[cls])[indices[:self.k_shot]].tolist())
            query_batch.append(np.array(self.data[cls])[indices[self.k_shot:]].tolist())

        # flatten 文件名列表
        flat_support = [os.path.join(self.img_path, fname)
                        for task in support_batch for fname in task]
        flat_query = [os.path.join(self.img_path, fname)
                      for task in query_batch for fname in task]
        
        # 根据文件名前缀提取原始标签（假设 CSV 中记录的 label 为文件名的前缀）
        orig_support = [self.img2label[fname.split('.')[0][:9]]
                        for task in support_batch for fname in task]
        orig_query = [self.img2label[fname.split('.')[0][:9]]
                      for task in query_batch for fname in task]

        # 将原始标签映射为相对标签（0~n_way-1）
        unique = np.unique(orig_support)
        rel_support = np.zeros_like(orig_support)
        rel_query = np.zeros_like(orig_query)
        for idx, lab in enumerate(unique):
            rel_support[np.array(orig_support) == lab] = idx
            rel_query[np.array(orig_query) == lab] = idx

        # 加载图像
        for i, path in enumerate(flat_support):
            support_imgs[i] = self.transform(path)
        for i, path in enumerate(flat_query):
            query_imgs[i] = self.transform(path)

        return support_imgs, torch.LongTensor(rel_support), query_imgs, torch.LongTensor(rel_query)

    def __len__(self):
        return self.batchsz
