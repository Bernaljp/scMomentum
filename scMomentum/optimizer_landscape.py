import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

class MaskedLinearLayer(nn.Module):
    """
    Linear layer with masked weights.
    """
    def __init__(self, input_size, output_size, mask, device, pre_initialized_W=None):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False, device=device)
        self.register_buffer('mask', torch.tensor(mask, dtype=torch.float32, device=device))

        n_in_mask = (mask.sum(dim=0) > 0).sum().sqrt()

        if pre_initialized_W is None:
            nn.init.uniform_(self.linear.weight, -1 / n_in_mask, 1 / n_in_mask)
        else:
            self.linear.weight = nn.Parameter(torch.tensor(pre_initialized_W, dtype=torch.float32, device=device))

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
    A model that learns:  output = W(s) + I - clamp(gamma, 0) * x
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
        pre_initialized_W: torch.Tensor = None,
        pre_initialized_I: torch.Tensor = None,
    ):
        super().__init__()
        self.device = device

        g = torch.log(torch.tensor(g, dtype=torch.float32, device=device)+1e-8)
        self.gamma = nn.Parameter(g) if refit_gamma else g

        scaffold = torch.tensor(scaffold, dtype=torch.float32, device=device)
        self.register_buffer("scaffold_raw", scaffold)

        scaffold_tfs = torch.zeros_like(scaffold)
        with torch.no_grad():
            scaffold_tfs[:, scaffold.sum(dim=0) > 0] = 1
        self.register_buffer("scaffold", scaffold_tfs)

        self.scaffold_lambda = scaffold_regularization
        n = g.shape[0]

        init_I = torch.rand((n,), device=device) if pre_initialized_I is None else torch.tensor(pre_initialized_I, dtype=torch.float32, device=device)
        self.I = nn.Parameter(init_I)

        if use_masked_linear:
            self.W = MaskedLinearLayer(n, n, self.scaffold, device=device, pre_initialized_W=pre_initialized_W)
        else:
            self.W = nn.Linear(n, n, bias=False, device=device)
            if pre_initialized_W is not None:
                self.W.weight = nn.Parameter(torch.tensor(pre_initialized_W, dtype=torch.float32, device=device))
            else:
                nn.init.xavier_uniform_(self.W.weight)

    def forward(self, inputs):
        """
        Args:
            inputs (tuple): (s, x), both shape (batch_size, n)
        Returns:
            torch.Tensor: Output of shape (batch_size, n)
        """
        s, x = inputs
        gamma_clamped = torch.exp(torch.clamp(self.gamma, max=10.0))
        # I_clamped = torch.exp(torch.clamp(self.I, max=10.0))
        I_clamped = self.I
        return self.W(s) + I_clamped - gamma_clamped * x

    def train_model(
        self,
        train_loader: DataLoader,
        epochs: int = 1000,
        learning_rate: float = 0.001,
        criterion: str = "L1",
        scheduler_fn=None,
        scheduler_kwargs={},
        get_plots=False,
        display_epoch=100,
    ):
        if scheduler_kwargs is None:
            scheduler_kwargs = {}

        loss_mapping = {"L1": nn.L1Loss, "MSE": nn.MSELoss, "L2": nn.MSELoss}
        if criterion not in loss_mapping:
            raise ValueError(f"Invalid criterion: {criterion}. Choose from {list(loss_mapping.keys())}")
        loss_fn = loss_mapping[criterion]()

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = scheduler_fn(optimizer, **scheduler_kwargs) if scheduler_fn else None

        loss_history, reconstruction_loss_history = [], []

        mask_m = 1.0 - self.scaffold_raw

        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            epoch_loss, epoch_reconstruction_loss = 0.0, 0.0

            for (s_batch, x_batch), target in train_loader:
                s_batch, x_batch, target = s_batch.to(self.device), x_batch.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self((s_batch, x_batch))

                reconstruction_loss = loss_fn(output, target)
                graph_constr_loss = self.scaffold_lambda * ((self.W.weight * mask_m).norm(2) + (self.W.weight * mask_m).norm(1))
                bias_loss = (torch.abs(self.I) + 10).norm(2)
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

            if (epoch % display_epoch == 0) or (epoch == epochs - 1):
                tqdm.write(f"[Epoch {epoch+1}/{epochs}] "
                           f"Total Loss: {avg_loss:.6f}, "
                           f"Reconstruction Loss: {avg_reconstruction_loss:.6f}, "
                           f"Batch size: {s_batch.shape[0]}")

                if get_plots:
                    _,axs = plt.subplots(1,2, figsize=(10,5))

                    axs[0].scatter(output[0].detach().cpu().numpy().flatten(), target[0].detach().cpu().numpy().flatten())
                    x_min, x_max = axs[0].get_xlim()
                    y_min, y_max = axs[0].get_ylim()
                    min_val = min(x_min, y_min)
                    max_val = max(x_max, y_max)
                    axs[0].plot([min_val, max_val], [min_val, max_val], color='k', linestyle='--', linewidth=1)

                    axs[1].imshow(self.W.weight.detach().cpu().numpy())
                    plt.show()

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