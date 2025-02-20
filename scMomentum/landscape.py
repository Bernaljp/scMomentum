import anndata
from collections import defaultdict
import dynamo as dyn
import hoggorm as ho
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy.spatial.distance import cdist
import seaborn as sns
import torch
from tqdm import tqdm
from typing import Union, List
import umap  # Ensure UMAP is imported, consider handling ImportError for environments without UMAP
from .utilities import *  # Make sure to import the utilities module for using utility functions
from .optimizer_landscape import ScaffoldOptimizer, CustomDataset
from .ode_solver import ODESolver

class Landscape:
    def __init__(self,
                 data: anndata.AnnData,
                 spliced_matrix_key: str = 'Ms',
                 velocity_key: str = 'velocity_S',
                 degradation_key: str = 'gamma',
                 genes: Union[None, List[str], List[bool], List[int]] = None,
                 cluster_key: Union[None, str] = None,
                 w_threshold: float = 1e-5,
                 w_scaffold: Union[None, np.ndarray] = None,
                 scaffold_regularization: float = 1.0,
                 only_TFs: bool = False,
                 infer_I: bool = False,
                 refit_gamma: bool = False,
                 pre_initialize_W: bool = False,
                 criterion: str = 'L2',
                 batch_size: int = 64,
                 n_epochs: int = 1000,
                 device: str = 'cpu',
                 use_scheduler: bool = False,
                 scheduler_kws: dict = {},
                 get_plots: bool = False,
                 skip_all: bool = False,
                 manual_fit: bool = False,):
        
        self.adata = data
        self.spliced_matrix_key = spliced_matrix_key
        self.velocity_key = velocity_key
        self.gamma_key = degradation_key
        self.genes = self.gene_parser(genes)  # Ensure this method exists to parse gene identifiers
        self.gene_names = self.adata.var.index[self.genes]
        self.cluster_key = cluster_key
        self.clusters = self.adata.obs[self.cluster_key].unique() if self.cluster_key else []
        self.scaffold = w_scaffold
        self.refit_gamma = refit_gamma

        if not manual_fit:
            # Fit sigmoids and heavysides for all genes
            self.fit_all_sigmoids()  # Adjust 'th' as necessary
            self.write_sigmoids()
            # Fit interactions
            self.fit_interactions(w_threshold=w_threshold,
                                  w_scaffold=w_scaffold,
                                  scaffold_regularization=scaffold_regularization,
                                  only_TFs=only_TFs, 
                                  infer_I=infer_I,
                                  refit_gamma=refit_gamma,
                                  pre_initialize_W=pre_initialize_W,
                                  n_epochs=n_epochs,
                                  device=device,
                                  skip_all=skip_all,
                                  criterion=criterion,
                                  batch_size=batch_size,
                                  use_scheduler=use_scheduler,
                                  scheduler_kws=scheduler_kws,
                                  get_plots=get_plots,
                                  )

            # Compute energies and their correlations with gene expressions
            self.get_energies()
            self.energy_genes_correlation()

    def get_matrix(self, key, genes=None):
        """
        Retrieve a specific matrix from the AnnData object based on the given key.
        
        Parameters:
            key (str): Key for the desired matrix in the AnnData layers.
            genes (list, optional): List of gene names or indices to subset the matrix. Defaults to None.
        
        Returns:
            np.ndarray: The requested matrix, optionally subset by genes.
        """
        if genes is None:
            return self.adata.layers[key]
        else:
            # gene_indices = self.gene_parser(genes)  # Ensure this method is implemented to resolve gene identifiers to indices
            # return self.adata.layers[key][:, gene_indices]
            return self.adata.layers[key][:, genes]


    def write_property(self, key, value):
        """
        Write a property (value) to the AnnData object under the specified key, determining the appropriate location based on the shape of the value.

        Parameters:
            key (str): Key under which to store the value.
            value (np.ndarray): The value to be stored.
        """
        shape = np.shape(value)
        
        # Scalar or 1D array
        if len(shape) == 1:
            if shape[0] == self.adata.n_obs:
                self.adata.obs[key] = value
            elif shape[0] == self.adata.n_vars:
                self.adata.var[key] = value
            else:
                self.adata.uns[key] = value
        
        # 2D array
        elif len(shape) == 2:
            if shape[0] == self.adata.n_vars:
                if shape[1] == self.adata.n_vars:
                    self.adata.varp[key] = value
                else:
                    self.adata.varm[key] = value
            elif shape[0] == self.adata.n_obs:
                if shape[1] == self.adata.n_vars:
                    self.adata.layers[key] = value
                elif shape[1] == self.adata.n_obs:
                    self.adata.obsp[key] = value
                else:
                    self.adata.obsm[key] = value
            else:
                self.adata.uns[key] = value
        
        # Other
        else:
            self.adata.uns[key] = value

    def fit_all_sigmoids(self, min_th=0.05):
        """
        Fit sigmoid functions to gene expression data for all genes.

        Args:
            min_th (float): Threshold for zero expression in percentage of maximum expression of the each gene.
        """
        # Retrieve expression data for all genes
        x = to_numpy(self.get_matrix(self.spliced_matrix_key, genes=self.genes).T)

        # Apply the fitting function to each gene's expression data
        results = np.array([fit_sigmoid(g, min_th=min_th) for g in x])

        # Unpack fitting results into separate attributes
        self.threshold, self.exponent, self.offset, self.sigmoid_mse = results.T

    def fit_interactions(
        self,
        w_threshold=1e-5,
        w_scaffold=None,
        scaffold_regularization=1.0,
        only_TFs=False,
        infer_I=False,
        refit_gamma=False,
        pre_initialize_W=False,
        n_epochs=1000,
        criterion="L2",
        batch_size=64,
        device="cpu",
        skip_all=False,
        use_scheduler=False,
        scheduler_kws={},
        get_plots=False,
    ):
        # Get spliced and velocity matrices
        x = to_numpy(self.get_matrix(self.spliced_matrix_key, genes=self.genes))
        v = to_numpy(self.get_matrix(self.velocity_key, genes=self.genes))
        g = self.adata.var[self.gamma_key][self.genes].values.astype(x.dtype)
        sig = self.get_matrix("sigmoid", genes=self.genes)
        
        self.W = {}
        self.I = {}
        if refit_gamma:
            self.gamma = {}
        
        clusters = self.adata.obs[self.cluster_key].unique() ##Maybe fix the case where cluster key fails
        if not skip_all:
            clusters = np.append(clusters, 'all') 
        if w_scaffold is not None:
            self.models = {}
        for ct in clusters:
            print(f"Inferring interaction matrix W and bias vector I for cluster {ct}")
            idx = self.adata.obs[self.cluster_key].values == ct
            self._fit_interactions_for_group(
                group=ct,
                x=x[idx, :],
                v=v[idx, :],
                sig=sig[idx, :],
                g=g,
                w_threshold=w_threshold,
                w_scaffold=w_scaffold,
                scaffold_regularization=scaffold_regularization,
                only_TFs=only_TFs,
                infer_I=infer_I,
                refit_gamma=refit_gamma,
                pre_initialize_W=pre_initialize_W,
                n_epochs=n_epochs,
                criterion=criterion,
                batch_size=batch_size,
                device=device,
                use_scheduler=use_scheduler,
                scheduler_kws=scheduler_kws,
                get_plots=get_plots,
            )


    def _fit_interactions_for_group(
        self,
        group,
        x,
        v,
        sig,
        g,
        w_threshold,
        w_scaffold,
        scaffold_regularization,
        only_TFs,
        infer_I,
        refit_gamma,
        pre_initialize_W,
        n_epochs,
        criterion,
        batch_size,
        device,
        use_scheduler,
        scheduler_kws,
        get_plots,
    ):
        """
        Helper function to fit interaction matrix W and bias vector I for a group (global or cluster).
        """
        device = torch.device("cuda" if (torch.cuda.is_available() and device == "cuda") else "cpu")

        W = None
        I = None
        if (w_scaffold is None) or pre_initialize_W: 
            # Default L2 criterion using least squares
            rhs = np.hstack((sig, np.ones((sig.shape[0], 1), dtype=x.dtype))) if infer_I else sig
            try:
                WI = np.linalg.lstsq(rhs, v + g[None, :] * x, rcond=1e-5)[0]
                W = WI[:-1, :].T if infer_I else WI
                I = WI[-1, :] if infer_I else -np.clip(WI, a_min=None, a_max=0).sum(axis=0)
            except:
                pass
        
        if w_scaffold is not None:
            # Use ScaffoldOptimizer
            model = ScaffoldOptimizer(g, w_scaffold, device, refit_gamma, scaffold_regularization=scaffold_regularization, use_masked_linear=only_TFs, pre_initialized_W=W, pre_initialized_I=I)
            train_loader = self._create_train_loader(sig, v, x, device, batch_size=batch_size)
            scheduler_fn = torch.optim.lr_scheduler.StepLR if use_scheduler else None
            scheduler_kwargs = {"step_size": 100, "gamma": 0.4} if scheduler_kws == {} else scheduler_kws
            model.train_model(train_loader, n_epochs, learning_rate=0.1, criterion=criterion, scheduler_fn=scheduler_fn, scheduler_kwargs=scheduler_kwargs, get_plots=get_plots)
            W = model.W.weight.detach().cpu().numpy()
            # I = np.exp(model.I.detach().cpu().numpy())
            I = model.I.detach().cpu().numpy()
            g = np.exp(model.gamma.detach().cpu().numpy())
            self.models[group] = model
            

        # Threshold values and store results
        W[np.abs(W) < w_threshold] = 0
        I[np.abs(I) < w_threshold] = 0
        self.W[group] = W
        self.I[group] = I
        if refit_gamma:
            self.gamma[group] = g
        


    def _create_train_loader(self, sig, v, x, device, batch_size=64):
        """
        Helper function to create a PyTorch DataLoader for training.
        """
        dataset = CustomDataset(sig, v, x, device)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def write_sigmoids(self):
        sig = self.get_sigmoid()
        sigmoids = np.zeros(list(self.adata.layers.values())[0].shape, dtype=sig.dtype)
        sigmoids[:, self.genes] = sig
        self.write_property('sigmoid', sigmoids)
        self.sigmoids = self.adata.layers['sigmoid']

    def get_energies(self, x=None):
        """
        Calculate and store the energies for each cluster or for a specific input x.
        
        Args:
            which (str): Specifies the type of activation function used in energy calculations. Default is 'sigmoid'.
            x (np.ndarray or None): If provided, calculates energies for this specific input instead of the entire dataset.
        
        Returns:
            If x is provided, returns a tuple of dictionaries (E, E_interaction, E_degradation, E_bias),
            otherwise, updates the class attributes with the calculated energies.
        """
        # Initialize dictionaries to store energies
        energies = {}
        interaction_energies = {}
        degradation_energies = {}
        bias_energies = {}

        # Iterate over each cluster to calculate energies
        for cluster in self.W.keys():
            # Calculate each component of the energy for the current cluster
            interaction_energy = self.interaction_energy(cluster, x=x)
            degradation_energy = self.degradation_energy(cluster, x=x)
            bias_energy = self.bias_energy(cluster, x=x)

            # Total energy is the sum of all components
            total_energy = interaction_energy + degradation_energy + bias_energy

            # Store the calculated energies
            interaction_energies[cluster] = interaction_energy
            degradation_energies[cluster] = degradation_energy
            bias_energies[cluster] = bias_energy
            energies[cluster] = total_energy

        # If x is None, update class attributes with the calculated energies
        if x is None:
            self.E = energies
            self.E_interaction = interaction_energies
            self.E_degradation = degradation_energies
            self.E_bias = bias_energies
        else:
            # If x is provided, return the calculated energies as a tuple of dictionaries
            return energies, interaction_energies, degradation_energies, bias_energies

    def write_energies(self):
        """
        Writes the calculated energies into the AnnData object as observations.
        """
        # Initialize energy columns in the AnnData observations with zeros
        self.adata.obs['Total_energy'] = np.zeros(self.adata.n_obs, dtype=float)
        self.adata.obs['Interaction_energy'] = np.zeros(self.adata.n_obs, dtype=float)
        self.adata.obs['Degradation_energy'] = np.zeros(self.adata.n_obs, dtype=float)
        self.adata.obs['Bias_energy'] = np.zeros(self.adata.n_obs, dtype=float)

        # Iterate over each cluster (excluding 'all') and update the energy values for cells in that cluster
        for cluster in [k for k in self.E if k != 'all']:
            # Identify the cells belonging to the current cluster
            cluster_indices = self.adata.obs[self.cluster_key] == cluster

            # Update energy values for cells in the current cluster
            self.adata.obs.loc[cluster_indices, 'Total_energy'] = self.E[cluster]
            self.adata.obs.loc[cluster_indices, 'Interaction_energy'] = self.E_interaction[cluster]
            self.adata.obs.loc[cluster_indices, 'Degradation_energy'] = self.E_degradation[cluster]
            self.adata.obs.loc[cluster_indices, 'Bias_energy'] = self.E_bias[cluster]

    def interaction_energy(self, cl, x=None):
        """
        Calculate the interaction energy for a given cluster or all cells.
        
        Args:
            cl (str): The cluster identifier or 'all' for all cells.
            x (np.ndarray, optional): Optional specific input to calculate energy for.
        
        Returns:
            np.ndarray: Calculated interaction energy.
        """
        # Determine the indices for the cluster or use all indices
        idx = self.adata.obs[self.cluster_key] == cl if cl != 'all' else slice(None)
        sig = self.get_sigmoid(x) if x is not None else self.get_matrix('sigmoid', genes=self.genes)[idx]
        W = self.W[cl]
        
        # Calculate the interaction energy
        interaction_energy = -0.5 * np.sum((sig @ W.T) * sig, axis=1)
        return interaction_energy

    def degradation_energy(self, cl, x=None):
        """
        Calculate the degradation energy for a given cluster or all cells.
        
        Args:
            cl (str): The cluster identifier or 'all' for all cells.
            x (np.ndarray, optional): Optional specific input to calculate energy for.
            which (str): Specifies the activation function used ('sigmoid' or other).
        
        Returns:
            np.ndarray: Calculated degradation energy.
        """
        idx = self.adata.obs[self.cluster_key] == cl if cl != 'all' else slice(None)
        g = self.adata.var[self.gamma_key][self.genes].values if not self.refit_gamma else self.gamma[cl]

        sig = self.get_sigmoid(x) if x is not None else self.get_matrix('sigmoid', genes=self.genes)[idx]
        integral = int_sig_act_inv(sig, self.threshold, self.exponent)
        degradation_energy = np.sum(g[None, :] * integral, axis=1)
        
        
        return degradation_energy

    def bias_energy(self, cl, x=None):
        """
        Calculate the bias energy for a given cluster or all cells.
        
        Args:
            cl (str): The cluster identifier or 'all' for all cells.
            x (np.ndarray, optional): Optional specific input to calculate energy for.
        
        Returns:
            np.ndarray: Calculated bias energy.
        """
        idx = self.adata.obs[self.cluster_key] == cl if cl != 'all' else slice(None)
        sig = self.get_sigmoid(x) if x is not None else self.get_matrix('sigmoid', genes=self.genes)[idx]
        I = self.I[cl]
        
        # Calculate the bias energy
        bias_energy = -np.sum(I[None, :] * sig, axis=1)
        return bias_energy

    def degradation_energy_decomposed(self, cl, x=None):
        """
        Calculate the decomposition of degradation energy for a given cluster or all cells.
        
        Args:
            cl (str): The cluster identifier or 'all' for all cells.
            x (np.ndarray, optional): Optional specific input to calculate energy for.
            which (str): Specifies the activation function used ('sigmoid' or other).
        
        Returns:
            np.ndarray: Decomposed degradation energy for each gene.
        """
        idx = self.adata.obs[self.cluster_key] == cl if cl != 'all' else slice(None)
        g = self.adata.var[self.gamma_key][self.genes].values if not self.refit_gamma else self.gamma[cl]

        sig = self.get_sigmoid(x) if x is not None else self.get_matrix('sigmoid', genes=self.genes)[idx]
        integral = int_sig_act_inv(sig, self.threshold, self.exponent)
        return g[None, :] * integral

    def bias_energy_decomposed(self, cl, x=None):
        """
        Calculate the decomposition of bias energy for a given cluster or all cells.
        
        Args:
            cl (str): The cluster identifier or 'all' for all cells.
            x (np.ndarray, optional): Optional specific input to calculate energy for.
        
        Returns:
            np.ndarray: Decomposed bias energy for each gene.
        """
        idx = self.adata.obs[self.cluster_key] == cl if cl != 'all' else slice(None)
        sig = self.get_sigmoid(x) if x is not None else self.get_matrix('sigmoid', genes=self.genes)[idx]
        I = self.I[cl]
        return -I[None, :] * sig
    
    def interaction_energy_decomposed(self, cl, side='in', x=None):
        """
        Calculate the decomposition of interaction energy for a given cluster or all cells.
        
        Args:
            cl (str): The cluster identifier or 'all' for all cells.
            side (str): Specifies the side of the interaction energy to decompose ('in' or 'out').
            x (np.ndarray, optional): Optional specific input to calculate energy for.
        
        Returns:
            np.ndarray: Decomposed interaction energy for each gene.
        """
        idx = self.adata.obs[self.cluster_key] == cl if cl != 'all' else slice(None)
        sig = self.get_sigmoid(x) if x is not None else self.get_matrix('sigmoid', genes=self.genes)[idx]
        W = self.W[cl]
        return -0.5*(sig @ W.T) * sig if side == 'out' else -0.5*(sig @ W) * sig
    
    def hopfield_model(self, cl, x=None):
        """
        Compute the Hopfield model for the dataset using the sigmoid activation function.

        Args:
            x (np.ndarray, optional): Input data for which to compute the Hopfield model.
                                    If None, the method uses the spliced matrix from the AnnData object.

        Returns:
            np.ndarray: The Hopfield model applied to the input data.
        """
        idx = self.adata.obs[self.cluster_key] == cl if x is None else slice(None)
        sig = self.get_sigmoid(x) if x is not None else self.get_matrix('sigmoid', genes=self.genes)[idx]
        W = self.W[cl]
        # Use the entire spliced matrix if x is not provided
        if x is None:
            x = to_numpy(self.get_matrix(self.spliced_matrix_key, genes=self.genes)[idx])

        g = self.adata.var[self.gamma_key][self.genes].values.astype(x.dtype) if not self.refit_gamma else self.gamma[cl]
        
        xdot = sig @ W.T - g[None, :] * x + self.I[cl]

        return xdot

        # Compute the Hopfield model using the sigmoid function values
        

        

    def jacobian_for_cell(self, cell):
        """
        Compute the Jacobian matrix for each point in x based on the model parameters.

        Args:
            x (np.ndarray): Array of points at which to compute the Jacobian. Each point represents a cell's state.

        Returns:
            List[np.ndarray]: List of Jacobian matrices for each point in x.
        """
        x = cell
        # Adjust the shape of exponent and threshold parameters based on the shape of x
        ex, th = (self.exponent, self.threshold) if x.ndim == 1 else (self.exponent[:, None], self.threshold[:, None])

        # Compute the sigmoid function values for x
        sig = self.get_sigmoid(x)

        # Retrieve the spliced matrix and compute distances to each point in x
        matrix = self.adata.layers[self.spliced_matrix_key][:, self.genes].toarray()
        dists = cdist(matrix, x)

        # Find the index of the closest cell in the dataset to each point in x
        minidx = np.argmin(dists, axis=0)

        # Determine the cell type for the closest cells
        celltype = self.adata.obs[self.cluster_key].values[minidx]

        # Compute the Jacobian for each point in x
        jacobians = []
        for i in range(len(x)):
            # Compute the derivative of the sigmoid function
            dsig_dx = np.nan_to_num(ex * sig[i] * (1 - sig[i]) / x[i])

            # Compute the Jacobian matrix using the interaction weights and the degradation rates
            W = self.W[celltype[i]]
            gamma = self.adata.var[self.gamma_key][self.genes].values.astype(sig.dtype) if not self.refit_gamma else self.gamma[celltype[i]]
            jacobian_matrix = W * dsig_dx[:, None] - np.diag(gamma)

            jacobians.append(jacobian_matrix)

        return jacobians

    def get_sigmoid(self, x=None):
        """
        Compute the sigmoid activation for the given input x or for the entire spliced matrix.

        Args:
            x (np.ndarray, optional): Input data for which to compute the sigmoid function.
                                    If None, the method uses the spliced matrix from the AnnData object.

        Returns:
            np.ndarray: The sigmoid activation applied to the input data.
        """
        # Use the entire spliced matrix if x is not provided
        if x is None:
            x = to_numpy(self.get_matrix(self.spliced_matrix_key, genes=self.genes))

        # Compute the sigmoid function of x using the class's threshold and exponent parameters
        sigmoid_output = np.nan_to_num(sigmoid(x, self.threshold[None, :], self.exponent[None, :]))
        

        return sigmoid_output

    def get_embedding(self, which='UMAP', **kwargs):
        """
        Compute an embedding for the dataset using UMAP or other specified methods.

        Args:
            which (str): The embedding method to use. Defaults to 'UMAP'.
            **kwargs: Additional keyword arguments to pass to the embedding method.
        """
        # Retrieve the spliced matrix data for the specified genes
        X = to_numpy(self.get_matrix(self.spliced_matrix_key, genes=self.genes))

        # Extract embedding parameters from kwargs or use defaults
        n_neighbors = kwargs.get('n_neighbors', 30)
        min_dist = kwargs.get('min_dist', 0.1)

        try:
            # Initialize the embedding object with specified parameters
            emb = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, 
                            low_memory=False, n_jobs=64, verbose=True)

            # Fit the embedding model to the data and transform it
            cells2d = emb.fit_transform(X)

            # Store the embedding model for future reference
            self.embedding = emb

            # Save the 2D coordinates of cells in the AnnData object
            self.adata.obsm[f'X_{which}'] = cells2d
        except ImportError:
            # Handle the case where UMAP is not installed
            print("UMAP is not installed. Please install it to use this functionality.")
        except Exception as e:
            # Handle other potential errors during embedding computation
            print(f"An error occurred while computing the {which} embedding: {e}")

    def energy_embedding(self, which='UMAP', resolution=50, **kwargs):
        """
        Compute and visualize the energy embedding for the dataset.
        
        Args:
            which (str): The embedding method used. Defaults to 'UMAP'.
            resolution (int): The resolution of the grid for energy computation. Higher values mean finer grids.
            **kwargs: Additional keyword arguments for the embedding method.
        """
        # Compute the embedding using the specified method
        self.get_embedding(which, **kwargs)
        
        # Initialize dictionaries to hold grid coordinates
        grid_X, grid_Y = {}, {}

        # Retrieve the 2D coordinates of cells from the embedding
        cells2d = self.adata.obsm[f'X_{which}']

        # Generate grids for energy computation for each cluster
        for k in self.W:
            # Determine cell indices for the current cluster or use all cells for 'all'
            cidx = self.adata.obs[self.cluster_key] == k if k != 'all' else np.arange(self.adata.n_obs)

            # Define the grid boundaries based on the min and max of the embedded coordinates
            minx, miny = np.min(cells2d[cidx], axis=0)
            maxx, maxy = np.max(cells2d[cidx], axis=0)
            grid_X[k], grid_Y[k] = np.mgrid[minx:maxx:resolution*1j, miny:maxy:resolution*1j]

        # Transform the grid points back to the high-dimensional space
        self.highD_grid = self.embedding.inverse_transform(np.vstack((grid_X.values(), grid_Y.values())).reshape(-1, 2))

        # Ensure the transformed points are non-negative
        self.highD_grid = np.maximum(self.highD_grid, 0)

        # Compute energies for the high-dimensional grid points
        E, inter, deg, bias = self.get_energies(x=self.highD_grid)

        # Initialize dictionaries to hold softened energies for each type
        Es, inters, degs, biases = {}, {}, {}, {}

        # Soften and reshape the computed energies for each cluster
        for i, k in enumerate(self.W):
            reshape_slice = slice(i * resolution**2, (i + 1) * resolution**2)
            Es[k] = soften(E[k][reshape_slice].reshape(grid_X[k].shape))
            inters[k] = soften(inter[k][reshape_slice].reshape(grid_X[k].shape))
            degs[k] = soften(deg[k][reshape_slice].reshape(grid_X[k].shape))
            biases[k] = soften(bias[k][reshape_slice].reshape(grid_X[k].shape))

        # Update the class attributes with the computed energy landscapes
        self.grid_X, self.grid_Y = grid_X, grid_Y
        self.grid_energy = Es
        self.grid_energy_interaction = inters
        self.grid_energy_degradation = degs
        self.grid_energy_bias = biases

        # Compute and store cell velocities in the AnnData object, with error handling for dyn
        dyn.tl.cell_velocities(self.adata, X=self.adata.layers[self.spliced_matrix_key], V=self.adata.layers[self.velocity_key], X_embedding=self.adata.obsm[f'X_{which}'], add_velocity_key=f'velocity_{which}')

    def save_embedding(self, filename):
        """
        Save the embedding and grid coordinates to a file using pickle.

        Args:
            filename (str): Path of the file where the embedding and grid coordinates will be saved.
        """
        # Create a dictionary containing the embedding and grid coordinates
        emb = {
            'embedding': self.embedding,
            'grid_X': self.grid_X,
            'grid_Y': self.grid_Y,
            'highD_grid': self.highD_grid
        }

        # Save the dictionary to the specified file
        with open(filename, 'wb') as outp:
            pickle.dump(emb, outp, pickle.HIGHEST_PROTOCOL)

    def load_embedding(self, filename, which='UMAP', resolution=50):
        """
        Load the embedding and grid coordinates from a file using pickle, and recalculate grid energies using previously computed high-dimensional grid points.

        Args:
            filename (str): Path of the file where the embedding and grid coordinates are saved.
            which (str): The key under which to store the loaded embedding in the AnnData object.
            resolution (int): The resolution used for grid generation in the energy embedding process.
        """
        # Load the dictionary containing the embedding and grid coordinates from the file
        with open(filename, 'rb') as inp:
            emb = pickle.load(inp)

        # Update the class attributes with the loaded embedding and grid coordinates
        self.embedding = emb['embedding']
        self.grid_X = emb['grid_X']
        self.grid_Y = emb['grid_Y']
        self.highD_grid = emb['highD_grid']

        # Transform cells to the embedding space to update the AnnData object
        X = to_numpy(self.get_matrix(self.spliced_matrix_key, genes=self.genes))
        cells2d = self.embedding.transform(X)
        self.adata.obsm[f'X_{which}'] = cells2d

        # Assume self.highD_grid is available and contains the high-dimensional grid points
        # Recalculate grid energies using the available high-dimensional grid points
        E, inter, deg, bias = self.get_energies(x=self.highD_grid)

        # Initialize dictionaries to hold softened energies for each cluster
        Es, inters, degs, biases = {}, {}, {}, {}

        # Soften and reshape the computed energies for each cluster
        for i, k in enumerate(self.W):
            reshape_slice = slice(i * resolution**2, (i + 1) * resolution**2)
            Es[k] = soften(E[k][reshape_slice].reshape(self.grid_X[k].shape))
            inters[k] = soften(inter[k][reshape_slice].reshape(self.grid_X[k].shape))
            degs[k] = soften(deg[k][reshape_slice].reshape(self.grid_X[k].shape))
            biases[k] = soften(bias[k][reshape_slice].reshape(self.grid_X[k].shape))

        # Update the grid energy attributes
        self.grid_energy = Es
        self.grid_energy_interaction = inters
        self.grid_energy_degradation = degs
        self.grid_energy_bias = biases

    def gene_parser(self, genes: Union[None, List[str], List[bool], List[int]]) -> np.ndarray:
        """
        Parses the given gene list into indices that indicate which genes to use.

        Args:
            genes (Union[None, List[str], List[bool], List[int]]): List of gene names, indices, or Boolean values, or None.

        Returns:
            np.ndarray: Array of indices indicating which genes to use.
        """
        if genes is None:
            # If genes is None, use all genes
            return np.arange(self.adata.n_vars)

        if isinstance(genes[0], str):
            # If the first element is a string, assume genes is a list of gene names
            gene_indices = self.adata.var.index.get_indexer_for(genes)
            # Check for -1 in gene_indices which indicates gene name not found
            if np.any(gene_indices == -1):
                missing_genes = np.array(genes)[gene_indices == -1]
                raise ValueError(f"Gene names not found in adata.var.index: {missing_genes}")
            return gene_indices
        elif isinstance(genes[0], (int,np.int64,np.int32,np.int16,np.int8)):
            # If the first element is an int, assume genes is a list of gene indices
            return np.array(genes)
        elif isinstance(genes[0], (bool, np.bool_)):
            # If the first element is a bool, assume genes is a Boolean mask
            if len(genes) != self.adata.n_vars:
                raise ValueError("Boolean gene list must have the same length as the number of genes in the dataset.")
            return np.where(genes)[0]
        else:
            raise ValueError("Genes argument must be None, a list of gene names, indices, or a Boolean mask.")

    def plot_fit(self, gene, color_clusters=False, ax=None, **fig_kws):
        """
        Plot the fit of a sigmoid or heavyside function to the expression data of a specified gene.

        Args:
            gene (str): The gene to plot.
            which (str): The type of fit to plot, either 'sigmoid' or another type like 'heavyside'.
            color_clusters (bool): If True, color points by cluster.
            ax (matplotlib.axes.Axes, optional): The axes on which to draw the plot. If None, a new figure and axes are created.
            **fig_kws: Additional keyword arguments passed to plt.subplots() when creating a new figure.

        Returns:
            matplotlib.axes.Axes: The axes containing the plot.
        """
        # Retrieve gene index and sort the expression data
        gene_index = self.gene_names.get_loc(gene)
        adata_index = self.genes[gene_index]
        gexp = to_numpy(self.get_matrix(self.spliced_matrix_key, genes=[adata_index])).flatten()
        expression_data = np.sort(gexp)
        empirical_cdf = np.linspace(0, 1, len(expression_data))

        # Initialize plot
        if ax is None:
            fig, ax = plt.subplots(1, 1, **fig_kws)
        ax.set_title(gene)

        # Plot original gene expression data
        if color_clusters:
            for cluster in self.adata.obs[self.cluster_key].unique():
                cluster_indices = self.adata.obs[self.cluster_key] == cluster
                ax.scatter(expression_data[cluster_indices], empirical_cdf[cluster_indices], label=f'Expression: {cluster}')
        else:
            ax.plot(expression_data, empirical_cdf, '.-', label='Gene expression', color=fig_kws.get('c1', 'k'))

        # Plot fitted curve
        threshold = self.threshold[gene_index]
        exponent = self.exponent[gene_index]
        offset = self.offset[gene_index]
        fitted_curve = sigmoid(expression_data, threshold, exponent) * (1 - offset) + offset


        ax.plot(expression_data, fitted_curve, '.-', label='Fit', color=fig_kws.get('c2', 'r'))

        # Add sigmoid formula text
        sigmoid_formula = r"$\frac{{x^{{{:.2f}}}}}{{x^{{{:.2f}}} + {:.2f}^{{{:.2f}}}}}$".format(exponent, exponent, threshold, exponent)
        ax.text(0.8, 0.4, sigmoid_formula, transform=ax.transAxes, fontsize=14)

        ax.legend(loc='lower right')
        return ax

    def plot_energy_boxplots(self, order=None, plot_energy = 'all', colors=None, **fig_kws):
        """
        Plot the energy distributions for different clusters using boxplots.

        Args:
            order (list, optional): Order of clusters to display in the boxplots.
            **fig_kws: Additional keyword arguments for plot customization.
        """
        if plot_energy == 'all':
            fig, axs = plt.subplots(2, 2, **fig_kws)
            axs[0, 0].set_title('Total Energy')
            axs[0, 1].set_title('Interaction Energy')
            axs[1, 0].set_title('Degradation Energy')
            axs[1, 1].set_title('Bias Energy')
            axs = axs.flatten()

            es = [self.E, self.E_interaction, self.E_degradation, self.E_bias]
        else:
            fig, axs = plt.subplots(1, 1, **fig_kws)
            axs = np.array([axs])
            es = [getattr(self, f'E_{plot_energy.lower()}')]

        if order is None:
            order = self.adata.obs[self.cluster_key].unique()
            if colors is not None:
                assert isinstance(colors, list) and len(colors) >= len(order), "Colors should be a list of length at least equal to the number of clusters."
                plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
        else:
            if colors is not None:
                assert (isinstance(colors, list) or isinstance(colors,dict)) and len(colors) >= len(order), "Colors should be a list of length at least equal to the number of clusters."
            colors = colors if isinstance(colors, dict) else {k: colors[i] for i, k in enumerate(order)}
            plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[colors[i] for i in order])
        
        for energy, ax in zip(es, axs):
            df = pd.DataFrame.from_dict(energy, orient='index').transpose().melt(var_name='Cluster', value_name='Energy').dropna()
            sns.boxplot(data=df, x='Cluster', y='Energy', order=order, ax=ax)
        
        plt.tight_layout()
        return axs

    def plot_energy_scatters(self, basis='umap', order=None, plot_energy = 'all', show_legend = False, **fig_kws):
        """
        Plot the energy landscapes for different clusters using 3D scatter plots.

        Args:
            basis (str): The basis used for embedding, default is 'umap'.
            **fig_kws: Additional keyword arguments for plot customization.
        """
        order = self.adata.obs[self.cluster_key].unique() if order is None else order
        
        if plot_energy == 'all':
            fig, axs = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, **fig_kws)
            # Set titles
            axs[0, 0].set_title('Total Energy')
            axs[0, 1].set_title('Interaction Energy')
            axs[1, 0].set_title('Degradation Energy')
            axs[1, 1].set_title('Bias Energy')

            axs = axs.flatten()
            es = [self.E, self.E_interaction, self.E_degradation, self.E_bias]
        else:
            fig, axs = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, **fig_kws)

            axs = np.array([axs])
            es = [getattr(self, f'E_{plot_energy.lower()}')]

        for k in order:
            cells = self.adata.obsm[f'X_{basis}'][self.adata.obs[self.cluster_key] == k][:, :2]
            for ax, energy_type in zip(axs, es):
                energies = energy_type[k]
                ax.scatter(*cells.T, energies, label=k)

        if show_legend:
            plt.legend()
            
        plt.tight_layout()

    def energy_genes_correlation(self):
        # Initialize an array to hold energies for all observations
        energies = np.zeros((4, self.adata.n_obs))
        
        # Initialize dictionaries to hold correlations for different energy types
        self.correlation = {}
        self.correlation_interaction = {}
        self.correlation_degradation = {}
        self.correlation_bias = {}

        # Loop through each cluster, including 'all'
        for k in self.W.keys():
            # Get indices of cells belonging to the current cluster or all cells if k is 'all'
            if k == 'all':
                continue
            else:
                cells = self.adata.obs[self.cluster_key] == k
            
            # Assign computed energies for the current cluster to the energies array
            energies[0, cells] = self.E.get(k, np.zeros(sum(cells)))
            energies[1, cells] = self.E_interaction.get(k, np.zeros(sum(cells)))
            energies[2, cells] = self.E_degradation.get(k, np.zeros(sum(cells)))
            energies[3, cells] = self.E_bias.get(k, np.zeros(sum(cells)))

            # Extract expression data for the current cluster or all cells
            X = to_numpy(self.adata.layers[self.spliced_matrix_key][cells][:, self.genes].T)

            # Compute correlations between energies and gene expression
            correlations = np.nan_to_num(np.corrcoef(np.vstack((energies[:, cells], X)))[:4, 4:])
            # Store computed correlations in their respective dictionaries
            self.correlation[k], self.correlation_interaction[k], self.correlation_degradation[k], self.correlation_bias[k] = correlations
        
        X = to_numpy(self.adata.layers[self.spliced_matrix_key][:,self.genes].T)
        correlations = np.nan_to_num(np.corrcoef(np.vstack((energies,X)))[:4,4:])
        self.correlation['all'], self.correlation_interaction['all'], self.correlation_degradation['all'], self.correlation_bias['all'] = correlations
        # No need to repeat the calculation for 'all', it's already covered in the loop
    
    def get_correlation_table(self, n_top_genes=20, which_correlation='total', order=None):
        corr = 'correlation_'+which_correlation.lower() if which_correlation.lower()!='total' else 'correlation'
        assert hasattr(self, corr), f'No {corr} attribute found in Landscape object'
        corrs_dict = getattr(self,corr)
        order = self.adata.obs[self.cluster_key].unique() if order is None else order
        df = pd.DataFrame(index=range(n_top_genes), columns=pd.MultiIndex.from_product([order, ['Gene', 'Correlation']]))
        for k in order:
            corrs = corrs_dict[k]
            indices = np.argsort(corrs)[::-1][:n_top_genes]
            genes = self.gene_names[indices]
            corrs = corrs[indices]
            df[(k, 'Gene')] = genes
            df[(k, 'Correlation')] = corrs
        return df

    def plot_high_correlation_genes(self, top_n=10, energy='total', cluster='all', absolute=False, basis='umap', plot_correlations=False, **fig_kws):
        # Determine the correct correlation dictionary based on the energy type
        if energy == 'interaction':
            corr_dict = self.correlation_interaction
        elif energy == 'degradation':
            corr_dict = self.correlation_degradation
        elif energy == 'bias':
            corr_dict = self.correlation_bias
        else:
            corr_dict = self.correlation

        # Get the correlations for the specified cluster
        corr = corr_dict[cluster]

        # Sort genes based on their absolute or relative correlation values
        abscorr = np.abs(corr) if absolute else corr
        top_genes_indices = np.argsort(abscorr)[-top_n:][::-1]
        top_genes_names = self.gene_names[top_genes_indices]
        plot = 'correlation' if plot_correlations else 'expression'
        self.plot_gene_correlation(top_genes_names, energy=energy, cluster=cluster, absolute=absolute, basis=basis, return_corr=False, plot=plot, **fig_kws)


    def plot_gene_correlation(self, genes, energy='total', cluster='all', absolute=False, basis='umap', return_corr=False, plot='correlation', **fig_kws):
        """
        Plot the correlation of a specified gene with the energy landscapes across one or more clusters.
        Optionally, the function can return the correlation values instead of plotting them.

        Parameters:
        - gene (str): The name of the gene for which the correlation with energy landscapes is to be plotted or returned.
        - energy (str): The type of energy landscape to consider for correlation. Options include 'total', 'interaction', 'degradation', and 'bias'. Default is 'total'.
        - clusters (Union[str, List[str]]): Specifies the clusters for which the correlation should be plotted or returned. Can be a single cluster name, a list of cluster names, or 'all' to consider all clusters. Default is 'all'.
        - absolute (bool): If True, considers the absolute value of correlations. Useful for highlighting strong negative correlations as well. Default is False.
        - basis (str): The embedding basis to be used for plotting. Typically, it would be 'umap', 'tsne', etc. Default is 'umap'.
        - return_corr (bool): If True, the function returns the correlation values instead of plotting them. Default is False.
        - plot (bool): If True, the function plots the correlation values using dyn.pl.scatters. If False, no plot is generated. Default is True.

        Returns:
        - If return_corr is True, returns a string representation of the gene's correlation with the energy landscapes for the specified clusters. Each line in the string represents a cluster and its corresponding correlation value.
        - If plot is True, generates a plot of the gene's correlation with the energy landscapes across the specified clusters and does not return anything.

        Note:
        - The function temporarily modifies the 'M_uu' layer of the AnnData object to store correlation values for plotting. It restores the 'M_uu' layer to its original state after plotting.
        """
        # Ensure clusters is a list for uniform processing
        clusts = self.E.keys()

        names = np.ravel([genes])
        gene_index = self.adata.var.index.get_indexer_for(names) if isinstance(names[0], str) else names
        correlation_values = {}

        for cl in clusts:
            if energy == 'interaction':
                corr = self.correlation_interaction[cl][self.gene_names.get_indexer_for(names)]
            elif energy == 'degradation':
                corr = self.correlation_degradation[cl][self.gene_names.get_indexer_for(names)]
            elif energy == 'bias':
                corr = self.correlation_bias[cl][self.gene_names.get_indexer_for(names)]
            else:
                corr = self.correlation[cl][self.gene_names.get_indexer_for(names)]

            correlation_values[cl] = corr

        # Backup and modify 'M_uu' layer
        M_uu_exists = 'M_uu' in self.adata.layers
        
        if M_uu_exists:
            backup_M_uu = self.adata.layers['M_uu'].copy()
        else:
            self.adata.layers['M_uu'] = np.zeros((self.adata.n_obs, self.adata.n_vars))

        for k,correl in correlation_values.items():

            row_indices = np.where(self.adata.obs[self.cluster_key] == k)[0]

            # Create a grid of indices using np.ix_
            grid_indices = np.ix_(row_indices, gene_index)

            # Use the grid to index into 'M_uu' and assign values
            self.adata.layers['M_uu'][grid_indices] = correl

        layer = 'M_uu' if plot.capitalize() == 'Correlation' else self.spliced_matrix_key
        # Plot correlations
        if plot in ['correlation', 'expression']:
            axs = dyn.pl.scatters(self.adata, basis=basis, color=genes, layer=layer, save_show_or_return='return', cmap='RdBu', vmin=-1, vmax=1, ncols=3, **fig_kws)
            axs = np.ravel([axs])

            cluster = self.W.keys() if cluster == 'all' else np.ravel([cluster])
            # Add titles to each subplot to indicate the cluster and correlation value
            titles = [f'{name}\n' for name in names]

            for clust in cluster:
                for i,co in enumerate(correlation_values[clust]):
                    titles[i] += f"{clust} correlation: {co:.3f}\n"
            # for cl, corr in correlation_values.items():
            #     for i,co in enumerate(corr):
            #         titles[i] += f"{cl} correlation: {co:.3f}\n"
            for ax,title in zip(axs, titles):
                ax.set_title(title)
            # ax.set_title(title)


        if M_uu_exists:
            self.adata.layers['M_uu'] = backup_M_uu
        else:
            del self.adata.layers['M_uu']
        
        if return_corr:
            # Return the collected correlation values as a string for all processed clusters
            return '\n'.join([f"{cl}-{g}: {corr:.3f}" for cl, cors in correlation_values.items() for g, corr in zip(names, cors)])

    def plot_gene_correlation_scatter(self, clus1, clus2, annotate=None, clus1_low=-0.5, clus1_high=0.5, clus2_low=-0.5, clus2_high=0.5, energy='total', ax=None):
        """
        Creates a scatter plot comparing the gene correlations with energy landscapes between two clusters.
        
        Parameters:
        - clus1 (str): The name of the first cluster for comparison.
        - clus2 (str): The name of the second cluster for comparison.
        - energy (str): Specifies the type of energy landscape ('total', 'interaction', 'degradation', 'bias') used for correlation. Default is 'total'.
        - ax (matplotlib.axes.Axes, optional): A matplotlib axes object to plot on. If None, a new figure and axes object are created.
        
        Returns:
        - matplotlib.axes.Axes: The axes object with the scatter plot.
        
        Note:
        - The function highlights correlations that are strongly positive in one cluster and strongly negative in the other, and vice versa, to identify genes with divergent behavior between the two clusters.
        - Dashed red lines at correlations of -0.5 and 0.5 serve as reference points to easily identify these divergent genes.
        """
        
        # Retrieve the correlation data for each cluster based on the specified energy type
        corr1 = getattr(self, f'correlation_{energy}')[clus1] if energy.lower() != 'total' else self.correlation[clus1]
        corr2 = getattr(self, f'correlation_{energy}')[clus2] if energy.lower() != 'total' else self.correlation[clus2]

        # Create a new figure and axes if none are provided
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
        else:
            fig = None

        # Set the limits for the axes
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))

        positions_corners = np.logical_or(np.logical_and(corr1 >= clus1_high, corr2 <= clus2_low), np.logical_and(corr1 <= clus1_low, corr2 >= clus2_high))

        # Identify correlations that are in opposite corners of the plot
        corr_corners = np.where(positions_corners)[0]

        # Identify correlations that are not in the opposite corners
        corr_center = np.where(~positions_corners)[0]

        # Plot the correlations using different colors for clarity
        ax.scatter(corr1[corr_corners], corr2[corr_corners], c='k', s=0.6, label='Divergent Correlations')
        ax.scatter(corr1[corr_center], corr2[corr_center], c='lightgray', s=0.5, label='Other Correlations')

        if annotate is not None:
            nn = annotate
            # Get top 6 genes with the highest absolute correlation values
            cor_indices = np.argsort((corr1[corr_corners])**2 + (corr2[corr_corners])**2)[-nn:]
            # Get the names of the top 6 genes with the highest absolute correlation values
            gois = self.gene_names[corr_corners][cor_indices]
            # Adding labels for the top 6 genes with the highest absolute correlation values
            arrow_dict = {"width": 0.5, "headwidth": 0.5, "headlength": 1, "color": "gray"}
            for gg,xx,yy in zip(gois,corr1[corr_corners],corr2[corr_corners]):
                rand_shift_1 = np.random.uniform(-0.08,0.08)
                rand_shift_2 = np.random.uniform(-0.08,0.08)
                ax.annotate(gg, xy=(xx, yy), xytext=(xx+rand_shift_1, yy+rand_shift_2), arrowprops=arrow_dict)

        # Add reference lines and labels
        ax.vlines([clus1_low, clus1_high], ymin=-1, ymax=1, linestyles='dashed', color='r')
        ax.hlines([clus2_low, clus2_high], xmin=-1, xmax=1, linestyles='dashed', color='r')
        ax.set_xlabel(clus1)
        ax.set_ylabel(clus2)
        # ax.legend()

        return ax

    def celltype_correlation(self, modified=True, all_genes=False):
        """
        Computes the correlation between cell types based on their current gene expression profiles.

        Parameters:
        - modified (bool): Indicates whether to use the modified RV coefficient for correlation. Default is True.
        - all_genes (bool): If True, considers all genes in the dataset. If False, only considers the genes specified in `self.genes`. Default is False.

        Updates:
        - self.cells_correlation: A DataFrame containing the pairwise correlation coefficients between cell types.
        """
        # Retrieve unique cell types from the data
        keys = self.adata.obs[self.cluster_key].unique()

        # Choose the correlation function based on the 'modified' parameter
        corr_f = ho.mat_corr_coeff.RV2coeff if modified else ho.mat_corr_coeff.RVcoeff

        # Initialize a DataFrame to hold the correlation coefficients
        rv = pd.DataFrame(index=keys, columns=keys, data=1.)

        # Determine the set of genes to consider
        genes_to_consider = None if all_genes else self.genes

        # Retrieve expression data for the chosen genes
        counts = self.get_matrix(self.spliced_matrix_key, genes=genes_to_consider)

        # Compute pairwise correlations between cell types
        for k1, k2 in itertools.combinations(keys, 2):
            expr_k1 = to_numpy(counts[self.adata.obs[self.cluster_key] == k1])
            expr_k2 = to_numpy(counts[self.adata.obs[self.cluster_key] == k2])
            rv.loc[k1, k2] = corr_f([expr_k1.T, expr_k2.T])[0, 1]
            rv.loc[k2, k1] = rv.loc[k1, k2]

        # Store the computed correlations
        self.cells_correlation = rv
    
    def plot_correlations_grid(self, colors=None, order=None, energy = 'total', x_low=-0.5, x_high=0.5, y_low=-0.5, y_high=0.5, **kwargs):
        """
        Plots a matrix where the diagonal shows cell types and the off-diagonal
        plots show gene correlation scatter plots between cell types.

        Args:
            cell_types (list): List of cell types to plot.
            colors (dict): Dictionary mapping cell types to colors.
            name (str): Name of the plot or dataset for title adjustments.
        """
        cell_types = self.adata.obs[self.cluster_key].unique() if order is None else order
        n = len(cell_types)
        figsize = kwargs.get('figsize', (15, 15))
        tight_layout = kwargs.get('tight_layout', True)
        fig, axs = plt.subplots(n, n, figsize=figsize, tight_layout=tight_layout)

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    for spine in axs[i, j].spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(2)
                        if colors is not None:
                            spine.set_color(colors[i])
                    # Remove ticks
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])
                    # Add text in the middle
                    text = cell_types[i]
                    text = text.replace(' ', '\n', 1)
                    text = text.replace('-', '-\n')
                    axs[i, j].text(0.5, 0.5, text, ha='center', va='center', fontsize=18, fontweight='bold', fontname='serif', transform=axs[i, j].transAxes)
                    
                    if colors is not None:
                        c = list(colors[cell_types[i]])
                        c[-1] = 0.2  # Assuming 'colors' values are RGBA
                        axs[i, j].set_facecolor(c)
                    continue
                
                axs[i, j].axis('off')
                self.plot_gene_correlation_scatter(clus1=cell_types[i], clus2=cell_types[j], energy=energy, ax=axs[j, i], clus1_low=x_low, clus1_high=x_high, clus2_low=y_low, clus2_high=y_high)
                # Remove ticks
                axs[j, i].set_xticks([])
                axs[j, i].set_yticks([])
                axs[j, i].set_xlabel('')
                axs[j, i].set_ylabel('')
                # Adjust ticks for the first column and last row
                if i == 0:
                    axs[j, i].set_yticks([-1, -0.5, 0, 0.5, 1])
                if j == n - 1:
                    axs[j, i].set_xticks([-1, -0.5, 0, 0.5, 1])

    def future_celltype_correlation(self, modified=True):
        """
        Computes the correlation between cell types based on their predicted future states.

        Parameters:
        - modified (bool): Indicates whether to use the modified RV coefficient for correlation. Default is True.

        Updates:
        - self.future_cells_correlation: A DataFrame containing the pairwise correlation coefficients between predicted future states of cell types.
        """
        # Retrieve unique cell types from the data
        keys = self.adata.obs[self.cluster_key].unique()

        # Choose the correlation function based on the 'modified' parameter
        corr_f = ho.mat_corr_coeff.RV2coeff if modified else ho.mat_corr_coeff.RVcoeff

        # Initialize a DataFrame to hold the correlation coefficients
        rv = pd.DataFrame(index=keys, columns=keys, data=1.)

        # Retrieve expression data for the specified genes
        counts = self.get_matrix(self.spliced_matrix_key, genes=self.genes)

        # Compute pairwise correlations between predicted future states of cell types
        for k1, k2 in itertools.combinations(keys, 2):
            future_expr_k1 = sigmoid(to_numpy(counts[self.adata.obs[self.cluster_key] == k1]), self.threshold[None, :], self.exponent[None, :])
            future_expr_k2 = sigmoid(to_numpy(counts[self.adata.obs[self.cluster_key] == k2]), self.threshold[None, :], self.exponent[None, :])

            # Apply interaction matrices to predict future states
            future_state_k1 = self.W[k1] @ future_expr_k1.T
            future_state_k2 = self.W[k2] @ future_expr_k2.T

            rv.loc[k1, k2] = corr_f([future_state_k1, future_state_k2])[0, 1]
            rv.loc[k2, k1] = rv.loc[k1, k2]

        # Store the computed correlations
        self.future_cells_correlation = rv

    def network_correlations(self):
        """
        Compute various correlations and distances between the interaction networks of different cell types.
        The interaction networks are represented by the interaction matrices W for each cell type.

        Updates:
        - self.jaccard: DataFrame containing Jaccard indices between cell types.
        - self.hamming: DataFrame containing Hamming distances between cell types.
        - self.euclidean: DataFrame containing Euclidean distances between cell types.
        - self.pearson: DataFrame containing Pearson correlations between cell types.
        - self.pearson_bin: DataFrame containing Pearson correlations between binary representations of cell types.
        - self.mean_col_corr: DataFrame containing mean column-wise Pearson correlations between cell types.
        - self.singular: DataFrame containing distances based on singular values between cell types.
        """
        # Retrieve unique cell types from the data
        keys = self.adata.obs[self.cluster_key].unique()

        # Initialize DataFrames to hold the computed metrics
        jaccard, hamming, pearson, pearson_bin, euclidean, mean_col, singular = \
            [pd.DataFrame(index=keys, columns=keys, data=d) for d in [1., 0., 1., 1., 0., 1., 0.]]

        # Compute singular values for each interaction matrix
        svs = {k: np.linalg.svd(self.W[k], compute_uv=False) for k in keys}

        # Compute pairwise metrics between cell types
        for k1, k2 in itertools.combinations(keys, 2):
            w1, w2 = self.W[k1], self.W[k2]
            bw1, bw2 = np.sign(w1), np.sign(w2)

            # Pearson correlation
            pearson.loc[k1, k2] = pearson.loc[k2, k1] = np.corrcoef(w1.ravel(), w2.ravel())[0, 1]

            # Pearson correlation for binary representations
            pearson_bin.loc[k1, k2] = pearson_bin.loc[k2, k1] = np.corrcoef(bw1.ravel(), bw2.ravel())[0, 1]

            # Euclidean distance
            euclidean.loc[k1, k2] = euclidean.loc[k2, k1] = np.linalg.norm(w1 - w2)

            # Hamming distance
            hamming.loc[k1, k2] = hamming.loc[k2, k1] = np.count_nonzero(bw1 != bw2)

            # Jaccard index
            intersection = np.logical_and(bw1, bw2)
            union = np.logical_or(bw1, bw2)
            jaccard.loc[k1, k2] = jaccard.loc[k2, k1] = intersection.sum() / union.sum()

            # Mean column-wise Pearson correlation
            mean_col_corr = np.mean(np.diag(np.corrcoef(w1, w2, rowvar=False)[:w1.shape[0], :w1.shape[0]]))
            mean_col.loc[k1, k2] = mean_col.loc[k2, k1] = mean_col_corr

            # Distance based on singular values
            singular.loc[k1, k2] = singular.loc[k2, k1] = np.linalg.norm(svs[k1] - svs[k2])

        # Store the computed metrics
        self.jaccard, self.hamming, self.euclidean, self.pearson, self.pearson_bin, self.mean_col_corr, self.singular = \
            jaccard, hamming, euclidean, pearson, pearson_bin, mean_col, singular
            

    def compute_jacobians(self, save_to_disk=False, save_dir=None, compute_eigenvectors=False, device='cpu'):

        """Set up storage for Jacobians, eigenvectors, and eigenvalues."""
        n_genes = len(self.genes)
        n_cells = self.adata.n_obs
        
        if save_to_disk:
            # Use np.memmap for disk-based storage
            self.jacobian = np.memmap(f'{save_dir}/jacobian.dat', dtype=np.float64, mode='w+', shape=(n_cells, n_genes, n_genes))
            self.jacobian_eigenvalues = np.memmap(f'{save_dir}/jacobian_eigenvalues.dat', dtype=np.complex128, mode='w+', shape=(n_cells, n_genes))
            if compute_eigenvectors:
                self.jacobian_eigenvectors = np.memmap(f'{save_dir}/jacobian_eigenvectors.dat', dtype=np.complex128, mode='w+', shape=(n_cells, n_genes, n_genes))
        else:
            # Use numpy arrays for memory-based storage
            self.jacobian = np.zeros((n_cells, n_genes, n_genes), dtype=np.float64)
            self.jacobian_eigenvalues = np.zeros((n_cells, n_genes), dtype=np.complex128)
            if compute_eigenvectors:
                self.jacobian_eigenvectors = np.zeros((n_cells, n_genes, n_genes), dtype=np.complex128)
        
        device = 'cuda' if ((device=='cuda') & torch.cuda.is_available()) else 'cpu'
        
        """Compute Jacobians and optionally eigenvectors for all clusters."""
        for cluster_label in self.W:
            # Skip the iteration if the cluster label is 'all'
            if cluster_label == 'all':
                continue

            g = self.adata.var[self.gamma_key][self.genes].values if not self.refit_gamma else self.gamma[cluster_label]
            gamma = torch.diag(torch.tensor(g.astype(list(self.W.values())[0].dtype), device=device))
            # Access the weight matrix for the current cluster and move it to the GPU
            W = torch.tensor(self.W[cluster_label], device=device)

            # Find the indices of cells belonging to the current cluster
            cluster_indices = np.where(self.adata.obs[self.cluster_key] == cluster_label)[0]

            # Extract the gene expression data for the cells in the current cluster
            cell_data = torch.tensor(to_numpy(self.adata.layers[self.spliced_matrix_key][cluster_indices][:, self.genes]), device=device)

            # Compute the sigmoid function and its first and second derivatives
            sigmoid_values = torch.tensor(self.get_sigmoid(cell_data.cpu().numpy()), device=device)
            sigmoid_prime = torch.tensor(self.exponent, device=device) * sigmoid_values * (1 - sigmoid_values) / ((1 - cell_data) * (cell_data == 0) + cell_data)

            # Update velocity fields and Jacobian matrices for each cell in the current cluster
            for idx, sig_prime_val in tqdm(zip(cluster_indices, sigmoid_prime), total=len(cluster_indices)):
                # Perform calculations on the GPU
                jac_f = W * sig_prime_val.view(-1,1) - gamma

                eigenvalues, eigenvectors = torch.linalg.eig(jac_f) if compute_eigenvectors else (torch.linalg.eigvals(jac_f), None)

                # Store the results
                self.jacobian_eigenvalues[idx] = eigenvalues.cpu().numpy()
                if compute_eigenvectors:
                    self.jacobian_eigenvectors[idx] = eigenvectors.cpu().numpy()
                self.jacobian[idx] = jac_f.cpu().numpy()

                # If saving to disk, flush data to persist it
                if save_to_disk:
                    self.jacobian_eigenvalues.flush()
                    if compute_eigenvectors:
                        self.jacobian_eigenvectors.flush()
                    self.jacobian.flush()


    def plot_jacobian_summary(self, fig_size=(15, 5), part='real', show=False):

        self.adata.obs['eval_positive_tmp'] = np.sum(np.real(self.jacobian_eigenvalues) > 0, axis=1)
        self.adata.obs['eval_negative_tmp'] = np.sum(np.real(self.jacobian_eigenvalues) < 0, axis=1)
        self.adata.obs['eval_mean_tmp'] = np.mean(np.real(self.jacobian_eigenvalues), axis=1) if part == 'real' else np.mean(np.imag(self.jacobian_eigenvalues[:,::2]), axis=1)

        fig,axs = plt.subplots(1,3,figsize=fig_size)
        dyn.pl.streamline_plot(self.adata, basis='umap', color='eval_positive_tmp', ax=axs[0], save_show_or_return='return')
        dyn.pl.streamline_plot(self.adata, basis='umap', color='eval_negative_tmp', ax=axs[1], save_show_or_return='return')
        dyn.pl.streamline_plot(self.adata, basis='umap', color='eval_mean_tmp', ax=axs[2], save_show_or_return='return')

        axs[0].set_title(f'Number of positive eigenvalues\n Jacobian')
        axs[1].set_title(f'Number of negative eigenvalues\n Jacobian')
        axs[2].set_title(f'Mean of {part} part of eigenvalues\n Jacobian')

        if show:
            plt.show()
        del self.adata.obs['eval_positive_tmp'], self.adata.obs['eval_mean_tmp'], self.adata.obs['eval_negative_tmp']

    def plot_jacobian_eigenvalue(self, n, part='real', ax=None, **kwargs):
        figsize = kwargs.get('figsize',(15,10))
        cmap = kwargs.get('cmap','viridis')
        _,ax = plt.subplots(1, 1, figsize=figsize) if ax is None else (None,ax)
        part = np.real if part=='real' else np.imag
        self.adata.obs[f'Eigenvalue {n+1}'] = part(to_numpy(self.adata.layers[f'jacobian_eigenvalues'][:,self.adata.var['use_for_dynamics'].values][:,n]).flatten())
        _ = dyn.pl.streamline_plot(self.adata, color=f'Eigenvalue {n+1}', basis='umap', size=(15,10), show_legend='on data', cmap=cmap, show_arrowed_spines=True, ax=ax, save_show_or_return='return')
        # plt.show()
        del self.adata.obs[f'Eigenvalue {n+1}']

    def plot_energy_surface_2d(self, clusters='all', energy='total', basis='UMAP', plot_cells=True, ax=None, **kwargs):
        """
        Plot a 2D contour plot of the energy surface, optionally overlaying cell locations.

        Parameters:
            clusters (str or list of str): Cluster(s) to be plotted. Use 'all' for all clusters.
            energy (str): Type of energy to plot ('total', 'interaction', 'degradation', 'bias').
            basis (str): Embedding basis to use for cell locations.
            plot_cells (bool): Whether to overlay cell locations on the plot.
            ax (matplotlib.axes.Axes): Matplotlib axis object for plotting. If None, a new figure is created.
            **kwargs: Additional keyword arguments for contour plotting and cell overlay.
        """
        # Energy selection based on the 'energy' argument
        energy_dict = {'total': self.grid_energy, 'interaction': self.grid_energy_interaction,
                       'degradation': self.grid_energy_degradation, 'bias': self.grid_energy_bias}
        energy_data = energy_dict.get(energy, self.grid_energy)  # Default to 'total' if energy type is not recognized

        if clusters == 'all':
            clusters = self.clusters

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the energy surface for each cluster
        for cluster in clusters:
            if cluster in energy_data:
                gX, gY = self.grid_X[cluster], self.grid_Y[cluster]
                E = energy_data[cluster]
                ax.contourf(gX, gY, E, levels=kwargs.get('levels', 20), cmap=kwargs.get('cmap', 'viridis'))

        # Optionally overlay cell locations
        if plot_cells:
            for cluster in clusters:
                if cluster != 'all':
                    cluster_idx = self.adata.obs['cluster_key'] == cluster
                    cells2d = self.adata.obsm[f'X_{basis}'][cluster_idx]
                    ax.scatter(cells2d[:, 0], cells2d[:, 1], label=cluster, edgecolor='k', linewidth=0.5, s=10)

        ax.set_title(f'{energy.capitalize()} Energy Surface')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.legend(title='Cluster')

        plt.colorbar(ax.contourf(gX, gY, E), ax=ax, label='Energy')

    def plot_energy_surface_3d(self, clusters='all', energy='total', basis='UMAP', plot_cells=True, ax=None, **kwargs):
        """
        Plot a 3D surface plot of the energy landscape, with the option to overlay cell locations.

        Parameters:
            clusters (str or list of str): Cluster(s) to be plotted. Use 'all' for all clusters.
            energy (str): Type of energy to plot ('total', 'interaction', 'degradation', 'bias').
            basis (str): Embedding basis to use for cell locations.
            plot_cells (bool): Whether to overlay cell locations on the plot.
            ax (matplotlib.axes._subplots.Axes3DSubplot): Matplotlib 3D axis object for plotting. If None, a new figure is created.
            **kwargs: Additional keyword arguments for surface plotting and cell overlay.
        """
        # Energy selection based on the 'energy' argument
        energy_dict = {'total': self.grid_energy, 'interaction': self.grid_energy_interaction,
                       'degradation': self.grid_energy_degradation, 'bias': self.grid_energy_bias}
        energy_data = energy_dict.get(energy, self.grid_energy)  # Default to 'total' if energy type is not recognized

        if clusters == 'all':
            clusters = self.clusters

        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        # Plot the energy surface for each cluster
        for cluster in clusters:
            if cluster in energy_data:
                gX, gY = self.grid_X[cluster], self.grid_Y[cluster]
                E = energy_data[cluster]
                ax.plot_surface(gX, gY, E, cmap=kwargs.get('cmap', 'viridis'), edgecolor='none', alpha=0.7)

        # Optionally overlay cell locations
        if plot_cells:
            for cluster in clusters:
                if cluster != 'all':
                    cluster_idx = self.adata.obs['cluster_key'] == cluster
                    cells2d = self.adata.obsm[f'X_{basis}'][cluster_idx]
                    cell_energy = self.rezet(gX, gY, E, cells2d)
                    ax.scatter(cells2d[:, 0], cells2d[:, 1], cell_energy, label=cluster, edgecolor='k', s=5)

        ax.set_title(f'{energy.capitalize()} Energy Landscape')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Energy')
        ax.legend(title='Cluster')

    def plot_gene_data(self, axis1, axis2,  energy_type, energy_type_2=None, top_n_axis1=1, top_n_axis2=1, order=None, interaction_direction=None, exclude_genes=[], show=False, **fig_kws):
        """
        Generalized plotting method for gene data.

        Parameters:
            order (list): List of cluster orders.
            energy_ths (list): List of energy thresholds for each cluster.
            energy_type (str): Type of energy ('Total', 'Interaction', 'Degradation', 'Bias').
            axis1 (str): Data to plot on x-axis ('rate', 'expression', 'energy').
            axis2 (str): Data to plot on y-axis ('rate', 'expression', 'energy').
        """
        order = self.adata.obs[self.cluster_key].unique() if order is None else order
        n_plots =  len(order) + 1
        interaction_direction = 'in' if interaction_direction is None else interaction_direction
        energy_type_2 = energy_type if energy_type_2 is None else energy_type_2

        # threshold_axis1 = {k: threshold_axis1 for k in order} if not isinstance(threshold_axis1, dict) else threshold_axis1
        # threshold_axis2 = {k: threshold_axis2 for k in order} if not isinstance(threshold_axis2, dict) else threshold_axis2
        figsize = fig_kws.pop('figsize', (10, 6))
        fig, axs = plt.subplots(2, n_plots//2, figsize=figsize, tight_layout=True, sharex=True)
        # Prepare data based on axis1 and axis2
        plot_data = {'rate':{}, 'expression':{}, 'energy':{}}
        plot_labels = {'rate':{}, 'expression':{}, 'energy':{}}

        genes_in = np.where(~np.isin(self.gene_names, exclude_genes))[0]

        if 'rate' in [axis1, axis2]:
            if (energy_type.lower() == 'interaction') or (energy_type_2.lower() == 'interaction'):
                plot_data['rate']['interaction'] = {k: self.W[k].sum(axis=0)[genes_in] for k in order} if interaction_direction.lower()=='in' else {k: self.W[k].sum(axis=1)[genes_in] for k in order}
                plot_labels['rate']['interaction'] = 'Total outgoing interaction rate' if interaction_direction.lower()=='out' else 'Total incoming interaction rate'
            if (energy_type.lower() == 'degradation') or (energy_type_2.lower() == 'degradation'):
                plot_data['rate']['degradation'] = {k: self.adata.var[self.gamma_key][self.genes].values[genes_in] if self.refit_gamma else self.gamma[k][genes_in] for k in order}
                plot_labels['rate']['degradation'] = 'Degradation rate'
            if (energy_type.lower() == 'bias') or (energy_type_2.lower() == 'bias'):
                plot_data['rate']['bias'] = {k: self.I[k][genes_in] for k in order}
                plot_labels['rate']['bias'] = 'Bias rate'
                
        if 'expression' in [axis1, axis2]:
            M = self.get_matrix(self.spliced_matrix_key)[:, self.genes[genes_in]]
            expression_dict = {}

            for k in order:
                cidxs = np.where(self.adata.obs[self.cluster_key]==k)[0]
                expression_dict[k] = to_numpy(np.mean(M[cidxs],axis=0)).squeeze()

            plot_data['expression'] = defaultdict(lambda: expression_dict)
            plot_labels['expression'] = defaultdict(lambda: 'Mean gene expression')
        
        if 'energy' in [axis1, axis2]:
            if energy_type.lower() == 'total':
                plot_data['energy']['total'] = {k: np.mean(self.interaction_energy_decomposed(k, interaction_direction)[:,genes_in]\
                                + self.degradation_energy_decomposed(k)[:,genes_in]\
                                + self.bias_energy_decomposed(k)[:,genes_in], axis=0) for k in order}
                plot_labels['energy']['total'] = 'Total energy'

            if (energy_type.lower() == 'interaction') or (energy_type_2.lower() == 'interaction'):
                plot_data['energy']['interaction'] = {k: np.mean(self.interaction_energy_decomposed(k, interaction_direction)[:,genes_in], axis=0) for k in order}
                plot_labels['energy']['interaction'] = 'Mean outgoing interaction energy' if interaction_direction.lower()=='out' else 'Mean incoming interaction energy'
            if (energy_type.lower() == 'degradation') or (energy_type_2.lower() == 'degradation'):
                plot_data['energy']['degradation'] = {k: np.mean(self.degradation_energy_decomposed(k)[:,genes_in], axis=0) for k in order}
                plot_labels['energy']['degradation'] = 'Mean degradation energy'
            if (energy_type.lower() == 'bias') or (energy_type_2.lower() == 'bias'):
                plot_data['energy']['bias'] = {k: np.mean(self.bias_energy_decomposed(k)[:,genes_in], axis=0) for k in order}
                plot_labels['energy']['bias'] = 'Mean bias energy'
        
        for k,ax in zip(order, axs.flat):
            
            axis1_data = plot_data[axis1][energy_type][k]
            axis2_data = plot_data[axis2][energy_type_2][k]

            ax.set_title(k)
            ax.set_xlabel(plot_labels[axis1][energy_type])
            ax.set_ylabel(plot_labels[axis2][energy_type_2])

            ax.scatter(axis1_data, axis2_data, color='gray', alpha=0.5, s=4)

            threshold_axis1 = np.sort(abs(axis1_data))[::-1][top_n_axis1]
            threshold_axis2 = np.sort(abs(axis2_data))[::-1][top_n_axis2]

            for x, y, gene in zip(axis1_data, axis2_data, self.gene_names):
                if (abs(x) > threshold_axis1) or (abs(y) > threshold_axis2):
                    ax.scatter(x, y, color='blue', s=1)
                    random_offset_x = np.random.uniform(-0.5, 1) * (max(axis1_data) - min(axis1_data)) * 0.1
                    random_offset_y = np.random.uniform(-0.5, 1) * (max(axis2_data) - min(axis2_data)) * 0.05
                    ax.annotate(gene, xy=(x, y), xytext=(x + random_offset_x, y + random_offset_y),
                                textcoords='data', fontsize=8, arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))
                
        if show:
            plt.show()
        else:
            return fig, axs

    def simulate_cell(self, cluster, x0=None, time=10, n_steps=100, noise=0, solver=None, clip=False):
        """
        Simulate the trajectory of a single cell over time within a specific cluster.

        Parameters:
            cluster (str): The cluster for which to simulate the cell trajectory.
            x0 (np.ndarray): Initial state of the cell. If None, a random cell from the cluster is used.
            time (float): Total simulation time.
            n_steps (int): Number of steps in the simulation.
            noise (float): Noise level to add to the simulation.
            solver (callable): ODE solver function to use for the simulation.
            clip (bool): Whether to clip the state variables to non-negative values.

        Returns:
            np.ndarray: Simulated cell trajectory over time.
            np.ndarray (optional): 2D embedding of the simulated trajectory, if an embedding is available.
        """
        # Retrieve interaction matrix, degradation rates, bias, thresholds, and exponents for the cluster
        W = self.W[cluster]
        g = self.adata.var[self.gamma_key][self.genes].values.astype(W.dtype) if self.refit_gamma else self.gamma[cluster].astype(W.dtype)
        I = self.I[cluster]
        k = self.threshold
        n = self.exponent

        # Initialize the ODE solver with the system parameters
        syst = ODESolver(W, g, I, k, n, solver, clipping=clip)

        # Select a random cell from the cluster or use the provided initial state
        c_idx = self.adata.obs[self.cluster_key] == cluster
        if x0 is None:
            r_idx = np.random.choice(np.where(c_idx)[0])
            cell = to_numpy(self.get_matrix(self.spliced_matrix_key, self.genes)[r_idx, :]).flatten()
        else:
            cell = x0

        # Simulate the system over time
        x_hist = syst.simulate_system(cell, time, n_steps, noise)

        # Transform the simulated trajectory to a 2D embedding if available
        if hasattr(self, 'embedding'):
            x_hist_2d = self.embedding.transform(np.clip(x_hist, 0, None).T)
            return x_hist, x_hist_2d

        return x_hist
