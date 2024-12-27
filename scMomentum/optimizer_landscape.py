import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class ScaffoldOptimizer(nn.Module):
    def __init__(self, g: torch.Tensor, scaffold: torch.Tensor, device: torch.device):
        """
        Initialize the ScaffoldOptimizer with the given parameters.

        Args:
            g (torch.Tensor): Target vector.
            scaffold (torch.Tensor): Scaffold matrix.
            device (torch.device): Device for computation (CPU/GPU).
        """
        super().__init__()
        self.device = device
        self.g = torch.tensor(g, dtype=torch.float32, device=device)
        self.scaffold = torch.tensor(scaffold, dtype=torch.float32, device=device)
        n = g.shape[0]

        # Parameters
        self.I = nn.Parameter(torch.rand((n,), dtype=torch.float32, device=device, requires_grad=True))
        self.W = nn.Parameter(
            torch.rand((torch.count_nonzero(self.scaffold),), dtype=torch.float32, device=device, requires_grad=True)
        )

    def forward(self, inputs):
        """
        Forward pass of the model.

        Args:
            inputs (tuple): Tuple containing s (input vector) and x (input features).

        Returns:
            torch.Tensor: Output of the model.
        """
        s, x = inputs
        WW = torch.zeros_like(self.scaffold, device=self.device)
        WW[self.scaffold == 1] = self.W
        return s @ WW + self.I - self.g * x

    def train_model(
        self,
        train_loader: DataLoader,
        epochs: int,
        learning_rate: float,
        reg_lambda: float,
        l1_regularization: bool = False,
        criterion: str = "L1",
        scheduler_fn=None,
    ):
        """
        Train the model with the provided dataset.

        Args:
            train_loader (DataLoader): Training data loader.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
            reg_lambda (float): Regularization strength.
            l1_regularization (bool): Use L1 (True) or L2 (False) regularization. Defaults to False.
            criterion (str): Loss criterion. Defaults to "L1".
            scheduler_fn (function, optional): Learning rate scheduler function. Defaults to None.

        Returns:
            list: Training loss history.
        """
        # Select loss criterion dynamically
        try:
            loss_fn = getattr(nn, criterion + "Loss")()
        except AttributeError:
            raise ValueError(f"Invalid criterion: {criterion}")

        # Optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = scheduler_fn(optimizer,step_size=100,gamma=0.75) if scheduler_fn else None

        # Training loop
        loss_history = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for inputs, target in train_loader:
                inputs = (inputs[0].to(self.device), inputs[1].to(self.device))
                target = target.to(self.device)

                optimizer.zero_grad()
                output = self(inputs)
                loss = loss_fn(output, target)

                # Regularization
                reg_term = torch.tensor(0.0, device=self.device)
                for param in self.parameters():
                    reg_term += torch.norm(param, 1 if l1_regularization else 2)
                loss += reg_lambda * reg_term

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Update the learning rate if a scheduler is provided
            if scheduler:
                scheduler.step()

            # Log epoch details
            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)
            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        return loss_history


class OptimizerLandscape(nn.Module):
    def __init__(self, g, k, low_rank, device):
        super().__init__()
        n = g.shape[0]
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