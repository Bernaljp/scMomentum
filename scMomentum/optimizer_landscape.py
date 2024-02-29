import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

class OptimizerLandscape(nn.Module):
    def __init__(self, g, k, low_rank, device):
        super().__init__()
        n = g.shape[1]
        self.g = torch.tensor(g, dtype=torch.float32, device=device)
        k = k if low_rank else n

        self.init_sparse_mask(n, k, device, low_rank)
        self.init_parameters(n, k, device, low_rank)

    def init_sparse_mask(self, n, k, device, low_rank):
        self.sparse_mask_U = torch.zeros((n, k), dtype=torch.float32, device=device)
        for i in range(k):
            self.sparse_mask_U[torch.randperm(n)[:int(0.20 * n)], i] = 1
        if low_rank:
            self.sparse_mask_V = torch.zeros((k, n), dtype=torch.float32, device=device)
            for i in range(k):
                self.sparse_mask_V[i, torch.randperm(n)[:int(0.20 * n)]] = 1
        else:
            self.sparse_mask_V = None

    def init_parameters(self, n, k, device, low_rank):
        self.U = nn.Parameter(torch.rand((n, k), requires_grad=True, dtype=torch.float32, device=device) * self.sparse_mask_U)
        if low_rank:
            self.V = nn.Parameter(torch.rand((k, n), requires_grad=True, dtype=torch.float32, device=device) * self.sparse_mask_V)
        else:
            self.V = torch.eye(n, dtype=torch.float32, device=device, requires_grad=False)
        self.I = nn.Parameter(torch.rand((n,), requires_grad=True, dtype=torch.float32, device=device))

    def forward(self, inputs):
        s, x = inputs
        return s @ self.U @ self.V + self.I - self.g * x

    def train_model(self, train_loader, epochs, learning_rate, reg_lambda, l1_regularization, criterion='L1'):
        criterion = nn.MSELoss() if criterion == 'MSE' else nn.L1Loss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            for inputs, target in train_loader:
                optimizer.zero_grad()
                output = self(inputs)
                loss = criterion(output, target)

                # Regularization
                l1_reg = torch.tensor(0., device=self.g.device)
                l2_reg = torch.tensor(0., device=self.g.device)
                for param in self.parameters():
                    if l1_regularization:
                        l1_reg += torch.norm(param, 1)
                    else:
                        l2_reg += torch.norm(param, 2)

                loss += reg_lambda * (l1_reg if l1_regularization else l2_reg)
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

class CustomDataset(Dataset):
    def __init__(self, s, v, x, device):
        self.s = torch.tensor(s, dtype=torch.float32, device=device)
        self.v = torch.tensor(v, dtype=torch.float32, device=device)
        self.x = torch.tensor(x, dtype=torch.float32, device=device)

    def __len__(self):
        return self.s.shape[0]

    def __getitem__(self, idx):
        return (self.s[idx], self.x[idx]), self.v[idx]