import torch
import numpy as np
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch import nn

import data

'''
'''
class CountPrediction(nn.Module):
    def __init__(self, cell_size, gene_size):
        super(CountPrediction, self).__init__()

        self.k = nn.Parameter(torch.zeros(gene_size))
        self.d = nn.Parameter(torch.zeros(gene_size))

        # Switching time and the u/s count at that time
        self.t0_g3 = nn.Parameter(torch.zeros(gene_size))
        self.u0_g3 = nn.Parameter(torch.zeros(gene_size))
        self.s0_g3 = nn.Parameter(torch.zeros(gene_size))
        
        self.t = nn.Parameter(torch.zeros(cell_size, gene_size))

    def S_trans(self, gene_index, t):
        return 1 / (1 + torch.exp(-self.k[gene_index] * (t[:, gene_index] - self.t0_g3[gene_index] - self.d[gene_index])))
    
    def predict_u(self, gene_index, alpha, beta, S):
        tilde_u = ((alpha / beta) * (1 - torch.exp(-beta * self.t[:, gene_index])) * (1 - S) +
                   (alpha / beta) * S +
                   (self.u0_g3[gene_index] * torch.exp(-beta * (self.t[:, gene_index] - self.t0_g3[gene_index])) - (alpha / beta)) * S)
        for i in range(len(tilde_u)):
            if torch.isnan(tilde_u[i]):
                print(f"coeff: {alpha[i]}, {beta[i]}, {S}, {self.t[i][gene_index]}, {self.u0_g3[gene_index]}, {self.t0_g3[gene_index]} ")
        
        return tilde_u
    
    def predict_s(self, gene_index, alpha, beta, gamma, S):
        tilde_s = (((alpha / gamma) * (1 - torch.exp(-gamma * self.t[:, gene_index])) +
                    (alpha / (gamma - beta)) * (torch.exp(-gamma * self.t[:, gene_index]) - torch.exp(-beta * self.t[:, gene_index]))) * (1 - S) +
                   (alpha / gamma) * S +
                   (beta * self.u0_g3[gene_index] / (gamma - beta) * (torch.exp(-gamma * (self.t[:, gene_index] - self.t0_g3[gene_index])) - torch.exp(-beta * (self.t[:, gene_index] - self.t0_g3[gene_index])))) * S)

        return tilde_s

    def predict(self, gene_index, out):
        # [num_cells, 3] 
        alpha = out[:, 0] 
        gamma = out[:, 1]  
        beta = out[:, 2]   

        with torch.no_grad():
            # should we use large or small value here?
            for i in range(len(beta)):
                if beta[i] == 0:
                    beta[i] += np.random.uniform(0.5, 1)
                if beta[i] == gamma[i]:
                    beta[i] += np.random.uniform(0.5, 1)
                if gamma[i] == 0:
                    gamma[i] += np.random.uniform(0.5, 1)

        # Compute S_trans for each cell
        S = self.S_trans(gene_index, self.t)

        # Compute tilde_u and tilde_s for each cell
        tilde_u = self.predict_u(gene_index, alpha, beta, S)
        tilde_s = self.predict_s(gene_index, alpha, beta, gamma, S)

        return tilde_u, tilde_s

    def forward(self, gene_index, out):
        return self.predict(gene_index, out)
    
'''
GAT Implementation
One GAT model for each gene, they are totally uncorrelated

Graph vertices: cells
Graph edges: similarity (xy distance, gene expression distance) between cells
    --option: kNN
    --option: ?
Input (vertices attributes): expression, u, s data for each cell
Output: alpha, beta, gamma for each cell
    --question: do we need t?
'''
class GAT(nn.Module):    
    def __init__(self, n_cell, n_gene, layers, heads):
        # 3 out channels, alpha beta gamma (t?)
        super(GAT, self).__init__()
        self.n_cell, self.n_gene = n_cell, n_gene

        self.layers = layers
        self.out_channel_size = 3
        self.heads = heads # number of attention heads
        self.in_channel_size = 2 # or 3?
        self.mid_channel_size = 32 * self.heads
        # could test using more middle layer / different sizes?

        # one module list for one gene
        self.convlist = [nn.ModuleList() for _ in range(self.n_gene)]
        self.bnslist = [nn.ModuleList() for _ in range(self.n_gene)]
        sizes = [self.in_channel_size] + [self.mid_channel_size] * layers + [self.out_channel_size]
        for i in range(self.n_gene):
            for j in range(self.layers + 1):
                self.convlist[i].append(GATConv(sizes[j], sizes[j+1], heads=(self.heads if j != self.layers else 1), concat=(j==self.layers), dropout=0.6))
                if j != layers:
                    self.bnslist[i].append(nn.BatchNorm1d(self.mid_channel_size))

        self.cp = CountPrediction(self.n_cell, self.n_gene)

    def predict(self, gene_data, gene_index):
        x, edge_index, edge_weight = gene_data.x, gene_data.edge_index, gene_data.edge_attr
        x = x.float()
        for i in range(self.layers + 1):
            x = F.elu(self.convlist[gene_index][i](x, edge_index))
            if i != self.layers:
                x = self.bnslist[gene_index][i](x)
                x = F.dropout(x, p=0.6, training=self.training)
        return x
    
    def forward(self, gene_data, gene_index):
        # alpha beta gamma's
        out = self.predict(gene_data, gene_index)
        tilde_u, tilde_s = self.cp(gene_index, out)
        return tilde_u, tilde_s

# Question - intuitive meaning of this loss?
# see VeloVi implementation at https://github.com/YosefLab/velovi/blob/main/velovi/_module.py#L488
def switch_loss(u_switch, s_switch, top_u, top_s):
    switch_loss = torch.sum((u_switch-top_u)**2 + (s_switch-top_s)**2, dtype=torch.float64)
    return switch_loss

def count_loss(pred_u, pred_s, u, s):
    count_loss = torch.mean((u-pred_u)**2 + (s-pred_s)**2, dtype=torch.float64)
    return count_loss

