# scMomentum

`scMomentum` is a Python package designed for analyzing and interpreting single-cell RNA sequencing (scRNA-seq) data. By leveraging state-of-the-art computational methods, `scMomentum` provides insights into gene regulatory networks, cell-type-specific interactions, and dynamic cellular states.

## Features

- **Landscapes Class**: Utilize the `Landscapes` class to model gene expression landscapes and infer gene regulatory networks.
- **Dynamic Analysis**: Simulate cell state transitions and predict future cell states using inferred networks.
- **Visualization**: Generate insightful visualizations, including energy landscapes, gene expression fits, and correlation plots.

## Installation

Install `scMomentum` directly from GitHub using pip:

```bash
pip install git+https://github.com/bernaljp/scMomentum.git
```

Ensure you have all the necessary dependencies installed by checking the `requirements.txt` file.

## Usage

Here's a quick example to get you started with `scMomentum`:

```python
import scMomentum as scm
import anndata

# Load your scRNA-seq data into an AnnData object
adata = anndata.read_h5ad('your_data.h5ad')

# Initialize the Landscapes class with your data
landscapes = scm.Landscapes(adata)

# Fit gene regulatory networks
landscapes.fit_interactions()

# Visualize the energy landscape for a specific gene
landscapes.plot_energy_surface(clusters=['Cluster1', 'Cluster2'], energy='total')
```

## Documentation

For more detailed usage and API documentation, please refer to the [scMomentum Documentation](#).

## Contributing

We welcome contributions to `scMomentum`! If you'd like to contribute, please fork the repository and create a pull request with your changes. For more information, see our [Contribution Guidelines](CONTRIBUTING.md).

## License

`scMomentum` is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

## Citation

If you use `scMomentum` in your research, please cite our paper:

> Author et al., "scMomentum: A Computational Framework for Unraveling Single-Cell Gene Regulatory Networks," Journal, Year.

## Support

If you encounter any issues or have questions, please file an issue on the GitHub [issue tracker](https://github.com/yourusername/scMomentum/issues).
