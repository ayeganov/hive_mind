import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass


@dataclass
class ConvGene:
    """Represents a single convolution operation in the genome"""
    kernel_size: int  # Size of kernel (square)
    kernel_values: torch.Tensor  # Actual kernel weights
    activation: str  # Type of activation function

    def mutate(self, mutation_power=0.1):
        """Mutate kernel values"""
        noise = torch.randn_like(self.kernel_values) * mutation_power
        self.kernel_values += noise


@dataclass
class PoolingGene:
    """Represents a pooling operation"""
    relative_size: float  # Pool size as fraction of input (0.0-1.0)
    pool_type: str  # 'max' or 'avg'


class ConvolutionalGenome:
    def __init__(self, output_size: int):
        """
        Args:
            output_size: Number of output values to produce
        """
        self.output_size = output_size
        self.conv_genes: list[ConvGene] = []
        self.pooling_genes: list[PoolingGene] = []

        # Calculate dimensions for final output
        self.out_h = int(np.floor(np.sqrt(output_size)))
        self.out_w = int(np.ceil(output_size / self.out_h))

        # Initialize with minimal structure
        self._add_initial_genes()


    def _add_initial_genes(self):
        # Add initial 3x3 convolution
        initial_kernel = torch.randn(1, 1, 3, 3)
        self.conv_genes.append(ConvGene(
            kernel_size=3,
            kernel_values=initial_kernel,
            activation='relu'
        ))

        # Add initial pooling and final pooling
        self.pooling_genes.append(PoolingGene(
            relative_size=0.5,
            pool_type='avg'
        ))

        # Add final pooling as a gene with exact output dimensions
        self.pooling_genes.append(PoolingGene(
            relative_size=-1,  # Special flag for final pooling
            pool_type='avg'
        ))


    def mutate(self):
        """Apply random mutations"""
        mutations = [
            (self._mutate_kernel_values, 0.3),
            (self._mutate_kernel_size, 0.1),
            (self._mutate_pool_size, 0.1),
            (self._mutate_pool_type, 0.1),
            (self._add_conv, 0.1),
            (self._add_pooling, 0.1),
            (self._remove_random_gene, 0.1)
        ]

        for mutation_func, prob in mutations:
            if np.random.random() < prob:
                mutation_func()


    def _mutate_kernel_values(self):
        """Mutate values of existing kernels"""
        for gene in self.conv_genes:
            gene.mutate()


    def _mutate_kernel_size(self) -> None:
        """Mutate size of a random kernel"""
        if self.conv_genes:
            gene = np.random.choice(self.conv_genes)
            new_size = gene.kernel_size + np.random.choice([-2, -1, 1, 2])
            new_size = max(2, min(7, new_size))
            if new_size != gene.kernel_size:
                gene.kernel_size = new_size
                gene.kernel_values = torch.randn(1, 1, new_size, new_size)


    def _mutate_pool_type(self) -> None:
        if self.pooling_genes:
            gene: PoolingGene = np.random.choice(self.pooling_genes)
            gene.pool_type = np.random.choice(["max", "avg"])


    def _mutate_pool_size(self) -> None:
        """Mutate relative pool size"""
        if len(self.pooling_genes) > 1:
            gene = np.random.choice(self.pooling_genes)
            pick_new_gene = lambda g: g.relative_size == -1

            while pick_new_gene(gene):
                gene = np.random.choice(self.pooling_genes)

            delta = np.random.uniform(-0.2, 0.2)
            gene.relative_size = max(0.1, min(0.9, gene.relative_size + delta))


    def _add_conv(self):
        """Add a new convolution operation"""
        kernel_size = np.random.choice([3, 5])
        kernel = torch.randn(1, 1, kernel_size, kernel_size)
        self.conv_genes.append(ConvGene(
            kernel_size=kernel_size,
            kernel_values=kernel,
            activation=np.random.choice(['relu', 'tanh'])
        ))


    def _add_pooling(self):
        """Add a new pooling operation"""
        insert_idx = len(self.pooling_genes) - 1
        new_gene = PoolingGene(
            relative_size=np.random.uniform(0.3, 0.7),
            pool_type=np.random.choice(['max', 'avg'])
        )
        self.pooling_genes.insert(insert_idx, new_gene)


    def _remove_random_gene(self):
        """Remove a random gene if there's more than one of its type"""
        if len(self.conv_genes) > 1:
            idx = np.random.randint(len(self.conv_genes))
            self.conv_genes.pop(idx)
        elif len(self.pooling_genes) > 1:
            idx = np.random.randint(len(self.pooling_genes) - 1)
            self.pooling_genes.pop(idx)


    def create_network(self) -> nn.Module:
        """Convert genome to PyTorch module"""
        return ConvolutionalNetwork(self)


class ConvolutionalNetwork(nn.Module):
    def __init__(self, genome: ConvolutionalGenome):
        super().__init__()
        self.genome = genome

        # Create conv layers
        self.conv_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()

        for gene in genome.conv_genes:
            conv = nn.Conv2d(1, 1, gene.kernel_size, padding='same')
            if gene.kernel_size > 10:
                print(f"{gene.kernel_size=}")
            conv.weight.data = gene.kernel_values
            self.conv_layers.append(conv)

            if gene.activation == 'relu':
                self.activation_layers.append(nn.ReLU())
            else:
                self.activation_layers.append(nn.Tanh())

        # Store pooling configurations
        self.pool_configs = []
        for gene in genome.pooling_genes:
            if gene.relative_size == -1:  # Final pooling
                self.pool_configs.append((
                    (genome.out_h, genome.out_w),
                    gene.pool_type
                ))
            else:
                self.pool_configs.append((
                    gene.relative_size,
                    gene.pool_type
                ))

    def forward(self, x):
        # Apply conv and pooling operations
        for conv, activ in zip(self.conv_layers, self.activation_layers):
            x = activ(conv(x))

        # Apply pooling operations
        for pool_config, pool_type in self.pool_configs:
            if isinstance(pool_config, tuple):  # Final pooling
                out_h, out_w = pool_config
                if pool_type == 'max':
                    x = nn.functional.adaptive_max_pool2d(x, (out_h, out_w))
                else:
                    x = nn.functional.adaptive_avg_pool2d(x, (out_h, out_w))
            else:  # Relative pooling
                h, w = x.shape[2:]
                out_h = max(1, int(h * pool_config))
                out_w = max(1, int(w * pool_config))

                if pool_type == 'max':
                    x = nn.functional.adaptive_max_pool2d(x, (out_h, out_w))
                else:
                    x = nn.functional.adaptive_avg_pool2d(x, (out_h, out_w))

        return x.view(x.size(0), -1)


def main():
    import time
    genome = ConvolutionalGenome(output_size=12)
    network = genome.create_network()

    patches_50 = torch.randn(32, 1, 50, 50)
    patches_123 = torch.randn(32, 1, 123, 123)

    start = time.perf_counter()
    out_50 = network(patches_50)
    out_123 = network(patches_123)
    end = time.perf_counter()

    print(f"Total time: {end - start} sec")

    print(f"Output shapes: {out_50.shape}, {out_123.shape}")

    genome.mutate()
    network = genome.create_network()
    out_50 = network(patches_50)
    print(f"After mutation output shape: {out_50.shape}")


if __name__ == "__main__":
    main()
