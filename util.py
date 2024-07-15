import numpy as np
import math
import pandas as pd
from matplotlib import cm 
from matplotlib import pyplot as plt
import scvelo as scv
from torch.utils.data import Dataset, DataLoader

'''
Input:   xy - [N x 2] array for the coordinate of points.
         cell_size - the cell size used for discretization.
Output:  xy_new - [N x 2] array for the discretized (integral) coordinate of points.
'''
def discretize_coord(xy, cell_size=32):
    # only transform the coordinates
    # get grid boundaries
    x_max, x_min = np.max(xy[:, 0]), np.min(xy[:, 0])
    y_max, y_min = np.max(xy[:, 1]), np.min(xy[:, 1])

    x_lb = x_min - 0.5 * cell_size
    xlen = math.ceil((x_max - x_lb) / cell_size)
    x_rb = x_lb + xlen * cell_size

    y_lb = y_min - 0.5 * cell_size
    ylen = math.ceil((y_max - y_lb) / cell_size)
    y_rb = y_lb + ylen * cell_size

    xy_new = []
    for [px, py] in xy:
        x = int((px - x_lb) // cell_size)
        y = int((py - y_lb) // cell_size)
        xy_new.append([x, y])
    return np.array(xy_new)

'''
Input:   adata - an adata object with restriction
         target - the target to take spatial derivative. Must be present in adata.layer
         gridname - the coordinate of cells. Must be present in adata.obsm
Output:  Dx, Dy, A - three [X x Y x d] tensors for the numerically estimated gradient, and averaged target. Might contain NaN.

This function assume that an entry in input A (d-dimensional) is either all NaN, or all not NaN.
'''
def numeric_gradient(adata, target='spliced', gridname='X_xy_loc_disc'):
    # first, calculate the average target in each grid cell
    # for target = 'X', pass in None.
    xy = adata.obsm[gridname]
    target = adata.to_df(target).to_numpy()
    N, d = target.shape
    xmax = np.max(xy[:, 0])
    ymax = np.max(xy[:, 1])
    A = np.nan * np.empty((xmax+1, ymax+1, d))
    count = np.zeros((xmax+1, ymax+1))

    for i in range(N):
        x, y = xy[i] # coord as integer
        if count[x][y] == 0: # or A[x][y] is nan
            count[x][y] += 1
            A[x][y] = target[i]
        else:
            count[x][y] += 1
            # get the average
            A[x][y] += (target[i] - A[x][y]) / count[x][y]

    # now, calculate the gradient
    Dx = np.nan * np.empty_like(A)
    Dy = np.nan * np.empty_like(A)
    # for each cell
    Dx_cell = []
    Dy_cell = []
    
    for i in range(N):
        x, y = xy[i]
        avail = 0
        diff = np.zeros(d)
        if y != 0 and (not np.isnan(A[x][y-1][0])):
            avail += 1
            diff += A[x][y] - A[x][y-1]
        if y != ymax and (not np.isnan(A[x][y+1][0])):
            avail += 1
            diff += A[x][y+1] - A[x][y]
        if avail != 0:
            # there are nearby cells
            Dy[x][y] = diff / avail
        Dy_cell.append(Dy[x][y])
        
        avail = 0
        diff = np.zeros(d) # reset those for calculating x direction
        if x != 0 and not (np.isnan(A[x-1][y][0])):
            avail += 1
            diff += A[x][y] - A[x-1][y]
        if x != xmax and not (np.isnan(A[x+1][y][0])):
            avail += 1
            diff += A[x+1][y] - A[x][y]
        if avail != 0:
            Dx[x][y] = diff / avail
        Dx_cell.append(Dx[x][y])
        
    return Dx, Dy, A, count, np.array(Dx_cell), np.array(Dy_cell)

class SpatialDataset(Dataset):
    def __init__(self, xy, s, u, transform=None):
        assert len(xy) == len(s) and len(s) == len(u)
        self.xy = xy
        self.s = s
        self.u = u
        self.transform = transform

    def __len__(self):
        return len(self.xy)
    
    def __getitem__(self, idx):
        sample = self.xy[idx], self.s[idx], self.u[idx]

        if self.transform:
            sample = self.transform(sample)
        return sample

class SpatialGradientDataset(Dataset):
    def __init__(self, xy, latent, grad, transform=None):
        assert (len(xy) == len(latent)) and len(latent) == len(grad)
        self.xy = xy
        self.latent = latent
        self.grad = grad
        self.transform = transform

    def __len__(self):
        return len(self.xy)
    
    def __getitem__(self, idx):
        sample = self.xy[idx], self.latent[idx], self.grad[idx]

        if self.transform:
            sample = self.transform(sample)
        return sample