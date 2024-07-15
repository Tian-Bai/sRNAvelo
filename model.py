import torch
import numpy as np
from torch_geometric import nn as gnn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch import nn
import torch_geometric
import math
from torch.utils.data import Dataset
import scanpy as sc
import scvelo as scv
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import rbf_kernel
from tqdm import tqdm

''' 
in-channels: number of genes * 2 = dimension of spliced/unspliced conbined
latent_dim: dimension of latent representation
need to train VAE first
'''
class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim=32, hidden_dims=[64, 32], beta=4):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.beta = beta

        all_dim = [in_channels] + hidden_dims + [latent_dim]
        
        # encoder
        # in_channel is used as a temporary variable here
        modules = []
        for i in range(len(all_dim) - 2):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=all_dim[i], out_features=all_dim[i+1]), 
                    nn.LeakyReLU()
                )
            )
        self.encoder = nn.Sequential(*modules)
        self.f_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.f_log_var = nn.Linear(hidden_dims[-1], latent_dim)

        # decoder
        modules = []
        for i in range(len(all_dim) - 1, 0, -1):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=all_dim[i], out_features=all_dim[i-1]),
                    nn.LeakyReLU()
                )
            )
        self.decoder = nn.Sequential(*modules)

    def encode(self, input):
        res = self.encoder(input)
        mu = self.f_mu(res)
        log_var = self.f_log_var(res)
        return mu, log_var
    
    def decode(self, z):
        res = self.decoder(z)
        return res
    
    def reparam(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparam(mu, log_var)
        return [self.decode(z), input, mu, log_var]
    
    def loss(self, recon, input, mu, log_var, mn_scale):
        recon_loss = F.mse_loss(recon, input)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recon_loss + self.beta * mn_scale * kl_loss
        return loss, recon_loss, kl_loss
    
''' 
in_channels: 2 + dimension of VAE low-dim representation
out_channels: dimension of gradient representation
'''
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_dims=[32, 32], out_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels

        all_dim = [in_channels] + hidden_dims + [out_channels]
        modules = []
        for i in range(len(all_dim) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=all_dim[i], out_features=all_dim[i+1]),
                    # batchnorm? dropout?
                    # nn.Dropout(p=0.1),
                    nn.BatchNorm1d(num_features=all_dim[i+1]),
                    nn.LeakyReLU()
                )
            )
        self.mlp = nn.Sequential(*modules)

    def forward(self, input):
        return self.mlp(input)

    def train_model(self, dataloader, device, lr=1e-2, epoch=500):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        for e in tqdm(range(epoch)):
            loss = 0
            for batch_idx, (xy_train, latent_train, grad_train) in enumerate(dataloader):
                xy_train = xy_train.double().to(device)
                latent_train = latent_train.double().to(device)
                grad_train = grad_train.double().to(device)
        
                pred_grad = self.forward(torch.cat((xy_train, latent_train), dim=1))
                loss_batch = F.mse_loss(grad_train, pred_grad)
                loss += loss_batch
                
                optimizer.zero_grad()
                loss_batch.backward()
                
                # gradient clipping
                torch.nn.utils.clip_grad_norm(self.parameters(), 1)
                optimizer.step()
            losses.append(loss.cpu().detach() / len(dataloader))
        return np.array(losses)

# is not symmetric!
def ZI_lp_loss(y_true, y_pred, p, gamma):
    assert y_true.shape == y_pred.shape
    N, d = y_true.shape

    abs_diff = torch.abs(y_true - y_pred)
    weighted_diff = torch.where(y_true == 0, gamma * abs_diff ** p, abs_diff ** p)
    return torch.mean(torch.sum(weighted_diff, dim=1))
    loss = 0

'''
in_channels: dimension of VAE low-dim representation
graph structure: from x-y data
out_channels: dimension of gradient representation
by default, use 1-head attention.
gamma: the discount for zero targets. This is for better prediction for zero-inflated data.
'''
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_dims=[32, 32], out_channels=32, dropout=0.5, p=2, gamma=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels
        self.p = p
        self.gamma = gamma
        
        all_dim = [in_channels] + hidden_dims 

        # GAT part
        modules = []
        for i in range(len(all_dim) - 1):
            modules.append((gnn.GATConv(all_dim[i], all_dim[i+1], heads=1, dropout=dropout), 'x, edge_index -> x'))
            modules.append(nn.ReLU())

        self.gat = gnn.Sequential('x, edge_index', modules)
        self.final_linear = nn.Linear(hidden_dims[-1], out_channels)

    def forward(self, x, edge_index):
        res = self.gat(x, edge_index)
        return self.final_linear(res)

    def predict(self, x, edge_index):
        self.eval()
        with torch.no_grad():
            return self.forward(x, edge_index)

    def train_model(self, xy_train, latent_train, grad_train, edge_index, device, lr=1e-2, epoch=500):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        grad_notNaN = torch.where(~torch.isnan(grad_train[:, 0]))[0]
        
        for e in tqdm(range(epoch)):
            xy_train = xy_train.double().to(device)
            latent_train = latent_train.double().to(device)
            grad_train = grad_train.double().to(device)
            edge_index = edge_index.to(device)
    
            pred_grad = self.forward(torch.cat((xy_train, latent_train), dim=1), edge_index)
            # loss = F.mse_loss(grad_train[grad_notNaN], pred_grad[grad_notNaN])
            loss = ZI_lp_loss(grad_train[grad_notNaN], pred_grad[grad_notNaN], self.p, self.gamma)
            
            optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm(self.parameters(), 1)
            optimizer.step()
            losses.append(loss.cpu().detach())
        return np.array(losses)

''' 
Two branches for Zero-inflated GAT.
'''
class ZeroProbBranch(nn.Module):
    def __init__(self, in_channels, hidden_dims=[32, 32], out_channels=32, dropout=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels

        all_dim = [in_channels] + hidden_dims
        modules = []
        for i in range(len(all_dim) - 1):
            modules.append((gnn.GATConv(all_dim[i], all_dim[i+1], heads=1, dropout=dropout), 'x, edge_index -> x'))
            modules.append(nn.ReLU())
        self.gat = gnn.Sequential('x, edge_index', modules)
        self.classifier = nn.Linear(hidden_dims[-1], out_channels)

    def forward(self, x, edge_index):
        res = self.gat(x, edge_index)
        res = self.classifier(res)
        return F.sigmoid(res)

class RegressionBranch(nn.Module):
    def __init__(self, in_channels, hidden_dims=[32, 32], out_channels=32, dropout=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels

        all_dim = [in_channels] + hidden_dims
        modules = []
        for i in range(len(all_dim) - 1):
            modules.append((gnn.GATConv(all_dim[i], all_dim[i+1], heads=1, dropout=dropout), 'x, edge_index -> x'))
            modules.append(nn.ReLU())
        self.gat = gnn.Sequential('x, edge_index', modules)
        self.regressor = nn.Linear(hidden_dims[-1], out_channels)

    def forward(self, x, edge_index):
        res = self.gat(x, edge_index)
        return self.regressor(res)

class ZeroInflatedGAT(nn.Module):
    def __init__(self, in_channels, hidden_dims=[32, 32], out_channels=32, dropout=0.5):
        super().__init__()
        self.classifier = ZeroProbBranch(in_channels, hidden_dims, out_channels, dropout)
        self.regressor = RegressionBranch(in_channels, hidden_dims, out_channels, dropout)

    def forward(self, x, edge_index):
        prob_zero = self.classifier(x, edge_index)
        res = self.regressor(x, edge_index)
        return prob_zero, res

    def predict(self, x, edge_index):
        self.eval()
        with torch.no_grad():
            prob_zero, res = self.forward(x, edge_index)
            is_zero = (prob_zero > 0.5).double()
            pred = (1 - is_zero) * res
            return pred

    def loss(self, y_true, prob_zero, res, threshold=0.01):
        zero_mask = (y_true < threshold).double()
        nonzero_mask = (y_true >= threshold).double()
        classification_loss = F.binary_cross_entropy(prob_zero, zero_mask)
        regression_loss = F.mse_loss(res * nonzero_mask, y_true)
        return classification_loss, regression_loss

    def train_model(self, xy_train, latent_train, grad_train, edge_index, device, alpha=0.2, threshold=0.01, lr=1e-2, epoch=500):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        c_losses, r_losses = [], []
        grad_notNaN = torch.where(~torch.isnan(grad_train[:, 0]))[0]
        
        for e in tqdm(range(epoch)):
            xy_train = xy_train.double().to(device)
            latent_train = latent_train.double().to(device)
            grad_train = grad_train.double().to(device)
            edge_index = edge_index.to(device)
    
            prob_zero, res = self.forward(torch.cat((xy_train, latent_train), dim=1), edge_index)

            c_loss, r_loss = self.loss(grad_train[grad_notNaN], prob_zero[grad_notNaN], res[grad_notNaN], threshold)
            loss = alpha * c_loss + (1 - alpha) * r_loss
            c_losses.append(c_loss.cpu().detach())
            r_losses.append(r_loss.cpu().detach())
            
            optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm(self.parameters(), 1)
            optimizer.step()
        return np.array(c_losses), np.array(r_losses)

''' 
KCN model.
'''
class KCN(nn.Module):
    # assume for now that label is 1d
    def __init__(self, train_coords, train_features, train_labels, n_neighbors, hidden_size, device):
        super().__init__()
        self.train_coords = train_coords
        self.train_features = train_features
        self.train_labels = train_labels
        
        self.n_neighbors = n_neighbors
        self.hidden_size = hidden_size
        self.device = device
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors).fit(train_coords)
        distances, self.train_neighbors = self.knn.kneighbors(None, return_distance=True)

        self.length_scale = np.median(distances.flatten())
        with torch.no_grad():
            self.graph_inputs = []
            for i in range(self.train_coords.shape[0]):
                att_graph = self.build_graph(self.train_coords[i], self.train_features[i], self.train_neighbors[i])
                self.graph_inputs.append(att_graph)

        input_dim = train_features.shape[1] + train_labels.shape[1] + 2 + 1  # 2 for the coord, 1 for the indicator
        out_dim = train_labels.shape[1]

        self.gnn = KCN_GAT(input_dim, hidden_size).to(self.device)
        self.final_linear = nn.Linear(hidden_size[-1], out_dim)

    def forward(self, coords, features, train_indices=None):
        if train_indices is not None:
            batch_inputs = [self.graph_inputs[i] for i in train_indices]
        else:
            neighbors = self.knn.kneighbors(coords, return_distance=False)
            with torch.no_grad():
                batch_inputs = [self.build_graph(coords[i], features[i], neighbors[i]) for i in range(len(coords))]
        batch_inputs = torch_geometric.data.Batch.from_data_list(batch_inputs).to(self.device)

        output = self.gnn(batch_inputs.x, batch_inputs.edge_index, batch_inputs.edge_attr)
        output = torch.reshape(output, [-1, (self.n_neighbors + 1), output.shape[1]])
        center_output = output[:, 0]
        pred = F.relu(self.final_linear(center_output))
        return pred

    def train_model(self, lr=1e-2, epoch=500):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        for e in tqdm(range(epoch)):
            # try full gradient descend now
            pred = self.forward(self.train_coords, self.train_features, np.arange(len(self.train_coords)))
            loss = F.mse_loss(pred, self.train_labels.to(self.device))
            optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm(self.parameters(), 1)
            optimizer.step()
            losses.append(loss.cpu().detach())
        return np.array(losses)
        
    def build_graph(self, coord, feature, neighbors):
        out_dim = self.train_labels.shape[1]

        # build the graph feature matrix
        y = torch.concat([torch.zeros([1, out_dim]), self.train_labels[neighbors]], axis=0)
        indicator = torch.zeros([neighbors.shape[0] + 1])
        indicator[0] = 1.0         # indicate that this is the one we want to predict
        features = torch.concat([feature[None, :], self.train_features[neighbors]], axis=0)
        all_coords = torch.concat([coord[None, :], self.train_coords[neighbors]], axis=0)
        graph_features = torch.column_stack([y, indicator, features, all_coords])
        
        # build the neighbor matrix
        kernel = rbf_kernel(all_coords.numpy(), gamma = 1 / (2 * self.length_scale ** 2))
        adj = torch.from_numpy(kernel)

        # create a graph
        nz = adj.nonzero(as_tuple=True)
        edges = torch.stack(nz, dim=0)
        edge_weights = adj[nz]

        attributed_graph = torch_geometric.data.Data(x=graph_features, edge_index=edges, edge_attr=edge_weights, y=None)
        return attributed_graph

class KCN_GAT(nn.Module):
    def __init__(self, input_dim, hidden_size, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.dropout = dropout

        all_dim = [input_dim] + hidden_size
        for i in range(len(all_dim) - 1):
            conv = gnn.GATConv(all_dim[i], all_dim[i+1], heads=1)
            self.add_module(f"layer{i}", conv)

    def forward(self, x, edge_index, edge_weight):
        for conv in self.children():
            # print(x.is_cuda, edge_index.is_cuda, edge_weight.is_cuda)
            # print(next(self.parameters()).is_cuda)
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
        return x

''' 
in_channel: dimension of VAE low-dim representation
out_channel: number of genes * 4 (4 gradients for each gene)
latent_dim: dimension of gradient representation
'''
class MixedGradientModel(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim=32):
        super(nn.Module, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        self.regression = MLP_regression(self.in_channels, out_channels=latent_dim)
        self.interpolation = GAT_interpolation(self.in_channels, out_channels=latent_dim)

        # for averaging
        self.alphas = torch.full(latent_dim, fill_value=0.5)
        self.final_linear = nn.Sequential(
            nn.Linear(latent_dim, out_channels),
            nn.ReLU()
        )

    def forward(self, x, edge_index):
        regre = self.regression(x)
        interp = self.interpolation(x, edge_index)
        out = regre * self.alphas + interp * (1 - self.alphas)
        return self.final_linear(out)