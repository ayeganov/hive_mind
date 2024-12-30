import pytest
import torch
import numpy as np

from hive_mind.exploring.ops import ConvolutionalGenome


def test_basic_initialization():
    """Test basic genome initialization"""
    genome = ConvolutionalGenome(output_size=12)

    assert len(genome.conv_genes) == 1
    assert len(genome.pooling_genes) == 2  # Regular pooling + final pooling
    assert genome.conv_genes[0].kernel_size == 3
    assert genome.pooling_genes[0].relative_size == 0.5
    assert genome.pooling_genes[1].relative_size == -1  # Final pooling
    assert genome.out_h * genome.out_w == 12


def test_network_output_shapes():
    """Test if network produces correct output shapes for various inputs"""
    genome = ConvolutionalGenome(output_size=12)
    network = genome.create_network()

    # Test various input sizes
    input_sizes = [(1, 1, 32, 32), (1, 1, 50, 50), (1, 1, 100, 100), (1, 1, 77, 77)]

    for size in input_sizes:
        x = torch.randn(size)
        out = network(x)
        assert out.shape == (size[0], 12), f"Failed for input size {size}"

def test_kernel_value_mutation():
    """Test if kernel values actually change during mutation"""
    genome = ConvolutionalGenome(output_size=12)
    original_kernel = genome.conv_genes[0].kernel_values.clone()

    genome._mutate_kernel_values()

    assert not torch.allclose(original_kernel, genome.conv_genes[0].kernel_values)


def test_pooling_operations():
    genome = ConvolutionalGenome(output_size=12)
    network = genome.create_network()
 
    assert len(network.pool_configs) == 2

    final_pool = network.pool_configs[-1]
    assert isinstance(final_pool[0], tuple)
    out_h, out_w = final_pool[0]
    assert out_h * out_w == 12

    x = torch.randn(1, 1, 50, 50)
    out = network(x)
    assert out.shape == (1, 12)


def test_pooling_type_mutation():
    genome = ConvolutionalGenome(output_size=12)
    original_pool_type = genome.pooling_genes[0].pool_type

    # Force mutations until pool type changes
    for _ in range(10):
        genome._mutate_pool_type()
        if any(gene.pool_type != original_pool_type for gene in genome.pooling_genes[:-1]):
            break

    # Verify network still works
    network = genome.create_network()
    x = torch.randn(1, 1, 50, 50)
    out = network(x)
    assert out.shape == (1, 12)


def test_non_square_output_sizes():
    """Test with output sizes that aren't perfect squares"""
    test_sizes = [6, 12, 15, 20]

    for size in test_sizes:
        genome = ConvolutionalGenome(output_size=size)
        network = genome.create_network()

        # Verify output dimensions multiply to give desired size
        final_pool = network.pool_configs[-1][0]
        assert final_pool[0] * final_pool[1] == size
        # Test network output
        x = torch.randn(1, 1, 50, 50)
        out = network(x)
        assert out.shape == (1, size)


def test_add_pooling_behavior():
    """Test adding new pooling genes preserves final pooling"""
    genome = ConvolutionalGenome(output_size=12)
    initial_pool_count = len(genome.pooling_genes)

    genome._add_pooling()

    assert len(genome.pooling_genes) == initial_pool_count + 1
    assert genome.pooling_genes[-1].relative_size == -1  # Final pooling should still be last
    assert genome.pooling_genes[-2].relative_size >= 0.3  # New pooling should have valid relative size


def test_kernel_size_mutation():
    """Test kernel size mutation"""
    genome = ConvolutionalGenome(output_size=12)
    original_size = genome.conv_genes[0].kernel_size

    # Force multiple mutations to ensure at least one change
    for _ in range(10):
        genome._mutate_kernel_size()
        if genome.conv_genes[0].kernel_size != original_size:
            break

    # Verify kernel dimensions match the new size
    new_size = genome.conv_genes[0].kernel_size
    assert genome.conv_genes[0].kernel_values.shape == (1, 1, new_size, new_size)


def test_pool_size_mutation():
    """Test pool size mutation stays within bounds"""
    genome = ConvolutionalGenome(output_size=12)

    # Test multiple mutations
    for _ in range(100):
        genome._mutate_pool_size()
        all_but_last = slice(0, -1)
        for pool_gene in genome.pooling_genes[all_but_last]:
            assert 0.1 <= pool_gene.relative_size <= 0.9

        last_pool_gene = genome.pooling_genes[-1]
        assert last_pool_gene.relative_size == -1, "Last pool gene got modified"

        network = genome.create_network()
        out_dims = network.pool_configs[-1][0]
        assert out_dims[0] * out_dims[1] == 12


def test_add_conv_mutation():
    """Test adding convolution layers"""
    genome = ConvolutionalGenome(output_size=12)
    initial_count = len(genome.conv_genes)

    genome._add_conv()

    assert len(genome.conv_genes) == initial_count + 1
    assert genome.conv_genes[-1].kernel_size in [3, 5]


# Edge Cases

def test_zero_dimensional_input():
    """Test handling of 0-dimensional input"""
    genome = ConvolutionalGenome(output_size=12)
    network = genome.create_network()

    with pytest.raises(RuntimeError):
        x = torch.randn(1, 1, 0, 0)
        network(x)


def test_single_pixel_input():
    """Test handling of 1x1 input"""
    genome = ConvolutionalGenome(output_size=12)
    network = genome.create_network()

    x = torch.randn(1, 1, 1, 1)
    out = network(x)
    assert out.shape == (1, 12)


def test_many_mutations():
    """Test stability with many mutations"""
    genome = ConvolutionalGenome(output_size=12)

    # Apply many mutations
    for _ in range(1000):
        genome.mutate()

        # Verify structure remains valid
        assert len(genome.conv_genes) >= 1
        assert len(genome.pooling_genes) >= 1

        # Test network still works
        network = genome.create_network()
        x = torch.randn(1, 1, 50, 50)
        out = network(x)
        assert out.shape == (1, 12)


# Breaking Tests

def test_extreme_input_sizes():
    """Test very large and very small inputs"""
    genome = ConvolutionalGenome(output_size=12)
    network = genome.create_network()

    # Test very large input
    x_large = torch.randn(1, 1, 10000, 10000)
    out = network(x_large)
    assert out.shape == (1, 12)

    # Test rectangular input
    x_rect = torch.randn(1, 1, 100, 50)
    out = network(x_rect)
    assert out.shape == (1, 12)

def test_numerical_stability():
    """Test numerical stability with extreme values"""
    genome = ConvolutionalGenome(output_size=12)
    network = genome.create_network()

    # Test very large values
    x_large = torch.ones(1, 1, 50, 50) * 1e6
    out = network(x_large)
    assert not torch.isnan(out).any()

    # Test very small values
    x_small = torch.ones(1, 1, 50, 50) * 1e-6
    out = network(x_small)
    assert not torch.isnan(out).any()

def test_rapid_mutation_sequence():
    """Test rapid sequence of mutations followed by network creation"""
    genome = ConvolutionalGenome(output_size=12)

    for _ in range(100):
        # Mutate multiple times before creating network
        for _ in range(10):
            genome.mutate()

        # Verify network still works
        network = genome.create_network()
        x = torch.randn(1, 1, 50, 50)
        out = network(x)
        assert out.shape == (1, 12)

@pytest.mark.skip()
def test_memory_leak():
    """Test for memory leaks during repeated mutation and network creation"""
    import gc
    import psutil

    process = psutil.Process()
    initial_memory = process.memory_info().rss

    # Create and mutate many genomes
    for _ in range(1000):
        genome = ConvolutionalGenome(output_size=12)
        for _ in range(10):
            genome.mutate()
            network = genome.create_network()
            x = torch.randn(1, 1, 50, 50)
            out = network(x)

        del genome
        del network
        del out
        gc.collect()

    final_memory = process.memory_info().rss
    memory_growth = final_memory - initial_memory

    # Allow for some memory growth but not excessive
    assert memory_growth < 1e8  # 100MB limit


def test_mutation_reproducibility():
    """Test if mutations with same random seed produce same results"""
    # Save initial random states
    initial_np_state = np.random.get_state()
    initial_torch_state = torch.get_rng_state()

    # Set seeds for first genome
    np.random.seed(42)
    torch.manual_seed(42)

    # Create and mutate first genome
    genome1 = ConvolutionalGenome(output_size=12)
    for _ in range(10):
        genome1.mutate()
    network1 = genome1.create_network()

    # Reset random states to initial values
    np.random.set_state(initial_np_state)
    torch.set_rng_state(initial_torch_state)
    np.random.seed(42)
    torch.manual_seed(42)

    # Create and mutate second genome
    genome2 = ConvolutionalGenome(output_size=12)
    for _ in range(10):
        genome2.mutate()
    network2 = genome2.create_network()

    # Use same input for both networks
    x = torch.randn(1, 1, 50, 50)
    out1 = network1(x)
    out2 = network2(x)

    assert torch.allclose(out1, out2)
