import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class MaskedLinearLayer(nn.Module):
    """
    A Linear layer that supports masking of its weights. The mask is applied
    to the gradients during backprop, effectively zeroing out the masked weights.
    """
    def __init__(self, input_size, output_size, mask, device):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False, device=device)
        self.register_buffer('mask', torch.tensor(mask, dtype=torch.float32, device=device))

        n_in_mask = (mask.sum(dim=0) > 0).sum().sqrt()
        nn.init.uniform_(self.linear.weight, -1 / n_in_mask, 1 / n_in_mask)
        with torch.no_grad():
            self.linear.weight *= self.mask

        self.weight = self.linear.weight
        self.linear.weight.register_hook(self._apply_mask)

    def _apply_mask(self, grad):
        return grad * self.mask

    def forward(self, x):
        return self.linear(x)


class ScaffoldOptimizer(nn.Module):
    """
    A model that learns the mapping:
        output = W(s) + I - clamp(gamma, 0) * x
    with a scaffold-based regularization on W.
    """
    def __init__(
        self,
        g: torch.Tensor,
        scaffold: torch.Tensor,
        device: torch.device,
        refit_gamma: bool = False,
        scaffold_regularization: float = 1.0,
        use_masked_linear: bool = False,
    ):
        """
        Args:
            g (torch.Tensor): Target vector of shape (n,).
            scaffold (torch.Tensor): Scaffold matrix of shape (n, n).
            device (torch.device): Device for computation (CPU or GPU).
            refit_gamma (bool): If True, gamma is a learnable parameter.
            scaffold_regularization (float): Regularization coefficient for W.
            use_masked_linear (bool): If True, use MaskedLinearLayer instead of nn.Linear.
        """
        super().__init__()
        self.device = device

        g = torch.tensor(g, dtype=torch.float32, device=device)
        if refit_gamma:
            self.gamma = nn.Parameter(torch.log(g.clone()+1e-6))
        else:
            self.register_buffer("gamma", g)

        scaffold = scaffold.T
        scaffold = torch.tensor(scaffold, dtype=torch.float32, device=device)
        self.register_buffer("scaffold_raw", scaffold)

        scaffold_binary = torch.zeros_like(scaffold)
        with torch.no_grad():
            scaffold_binary[:,scaffold.sum(dim=0) > 0] = 1
        self.register_buffer("scaffold", scaffold_binary)

        self.scaffold_lambda = scaffold_regularization
        n = g.shape[0]

        self.I = nn.Parameter(torch.rand((n,), dtype=torch.float32, device=device))

        if use_masked_linear:
            self.W = MaskedLinearLayer(n, n, self.scaffold, device=device)
        else:
            self.W = nn.Linear(n, n, bias=False, device=device)
            nn.init.xavier_uniform_(self.W.weight)

    def forward(self, inputs):
        """
        Args:
            inputs (tuple): A tuple (s, x) where each element is of shape (batch_size, n).
        Returns:
            torch.Tensor: The output of the model, shape (batch_size, n).
        """
        s, x = inputs
        gamma_clamped = torch.exp(torch.clamp(self.gamma, max=10.0))
        I_clamped = torch.exp(torch.clamp(self.I, max=10.0))
        return self.W(s) + I_clamped - gamma_clamped * x

    def train_model(
        self,
        train_loader: DataLoader,
        epochs: int = 1000,
        learning_rate: float = 0.001,
        criterion: str = "L1",
        scheduler_fn=None,
        scheduler_kwargs=None,
    ):
        """
        Args:
            train_loader (DataLoader): Yields ((s_batch, x_batch), target_batch) for training.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
            criterion (str): The loss function to use ("L1" or "MSE").
            scheduler_fn (callable, optional): A function returning a PyTorch scheduler instance.
            scheduler_kwargs (dict, optional): Keyword arguments for the scheduler.

        Returns:
            (list, list): A tuple of two lists containing the total loss history and
                          the reconstruction loss history per epoch.
        """
        if scheduler_kwargs is None:
            scheduler_kwargs = {}

        loss_mapping = {"L1": nn.L1Loss, "MSE": nn.MSELoss}
        if criterion not in loss_mapping:
            raise ValueError(f"Invalid criterion: {criterion}. Choose from {list(loss_mapping.keys())}")
        loss_fn = loss_mapping[criterion]()

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = scheduler_fn(optimizer, **scheduler_kwargs) if scheduler_fn else None

        loss_history = []
        reconstruction_loss_history = []

        mask_m = 1.0 - self.scaffold_raw

        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            epoch_loss = 0.0
            epoch_reconstruction_loss = 0.0

            for batch in train_loader:
                (s_batch, x_batch), target = batch
                s_batch = s_batch.to(self.device)
                x_batch = x_batch.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                output = self((s_batch, x_batch))
                reconstruction_loss = loss_fn(output, target)
                graph_constr_loss = self.scaffold_lambda * (self.W.weight * mask_m).norm(1)
                bias_loss = (torch.exp(self.I) + 10).norm(2)
                total_loss = reconstruction_loss + graph_constr_loss + bias_loss

                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                epoch_reconstruction_loss += reconstruction_loss.item()

            if scheduler is not None:
                scheduler.step()

            avg_loss = epoch_loss / len(train_loader)
            avg_reconstruction_loss = epoch_reconstruction_loss / len(train_loader)
            loss_history.append(avg_loss)
            reconstruction_loss_history.append(avg_reconstruction_loss)

            if (epoch % 100 == 0) or (epoch == epochs - 1):
                print(f"[Epoch {epoch+1}/{epochs}] "
                      f"Total Loss: {avg_loss:.6f}, "
                      f"Reconstruction Loss: {avg_reconstruction_loss:.6f}")

        return loss_history, reconstruction_loss_history

class CustomDataset(Dataset):
    def __init__(self, s, v, x, device):
        self.s = torch.tensor(s, dtype=torch.float32, device=device)
        self.v = torch.tensor(v, dtype=torch.float32, device=device)
        self.x = torch.tensor(x, dtype=torch.float32, device=device)

    def __len__(self):
        return self.s.shape[0]

    def __getitem__(self, idx):
        return (self.s[idx], self.x[idx]), self.v[idx]