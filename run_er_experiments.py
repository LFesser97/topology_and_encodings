from attrdict import AttrDict
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx, to_dense_adj, erdos_renyi_graph
from torch_geometric.nn import GATConv, global_mean_pool, GATv2Conv, GCNConv
import torch_geometric.transforms as T

from torch.nn import ModuleList
import torch.nn.functional as F

import time
import tqdm
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input
from preprocessing import rewiring, sdrf, fosr, digl, borf

import pickle
import wget
import zipfile
import os


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_layers, width, num_classes=2):
        super(GCN, self).__init__()
        self.convs = ModuleList()
        self.convs.append(GCNConv(num_features, width))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(width, width))
        self.lin = torch.nn.Linear(width, num_classes)

    def forward(self, data):
        """
        Forward pass of the GCN model.
        """
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, data.batch)
        x = F.relu(self.lin(x))
        return F.log_softmax(x, dim=1)


class ERExperiment:
    def __init__(self, training_n_range = [40, 50], test_n_range = [100, 150],
                 training_p = 0.5, test_p = 0.5, student_depth = 3, student_width = 64):
        self.training_n_range = training_n_range
        self.test_n_range = test_n_range
        self.training_p = training_p
        self.test_p = test_p
        self.student_depth = student_depth
        self.student_width = student_width
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.teacher_model = GCN(1, 3, student_width//2).to(self.device)
        self.student_model = GCN(1, student_depth, student_width).to(self.device)
        self.training_dataset = self.generate_er_dataset(self.training_n_range, self.training_p, self.teacher_model)
        self.test_dataset = self.generate_er_dataset(self.test_n_range, self.test_p, self.teacher_model)


    def generate_er_dataset(self, n_range, p, teacher_model):
        """
        Generate a dataset of Erdos-Renyi graphs with n_range and p.
        """
        dataset = []
        for n in range(n_range[0], n_range[1]):
            for _ in range(10):
                edge_index = erdos_renyi_graph(n, p)
                data = Data(edge_index=edge_index)
                data.x = torch.ones(n, 1)
                data.y = teacher_model(data).argmax().unsqueeze(0)
                dataset.append(data)
        return dataset
    
    def train_student(self, student_model, dataset, epochs=200, lr=0.01):
        """
        Train the student model on the dataset.
        """
        dataset.to(self.device)
        student_model.train()
        optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs):
            for data in dataset:
                optimizer.zero_grad()
                out = student_model(data)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch}, Loss {loss.item()}")
        self.student_model = student_model
    
    def evaluate_student(self, student_model, dataset):
        """
        Evaluate the student model on the dataset.
        """
        dataset.to(self.device)
        student_model.eval()
        correct = 0
        for data in dataset:
            out = student_model(data)
            pred = out.argmax().item()
            correct += int(pred == data.y.item())
        return correct / len(dataset)