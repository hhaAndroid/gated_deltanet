"""Unit tests for GatedDeltaNet.

Tests verify that:
1. GatedDeltaNet (non-varlen) produces correct outputs
2. GatedDeltaNetVarlen produces correct outputs
3. Both versions produce identical outputs for the same inputs
"""
import pytest
import torch

from gated_deltanet import GatedDeltaNetConfig, GatedDeltaNet, GatedDeltaNetVarlen


# Default test configuration
TEST_CONFIG = GatedDeltaNetConfig(
    hidden_size=128,
    num_key_heads=4,
    num_value_heads=4,
    key_head_dim=32,
    value_head_dim=32,
    conv_kernel_size=4,
    rms_norm_eps=1e-6,
    use_qk_l2norm=True,
    chunk_size=16,
)


class TestGatedDeltaNet:
    """Test basic GatedDeltaNet functionality."""
    
    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        config = GatedDeltaNetConfig(
            hidden_size=64,
            num_key_heads=2,
            num_value_heads=2,
            key_head_dim=32,
            value_head_dim=32,
            conv_kernel_size=4,
        )
        model = GatedDeltaNet(config, layer_idx=0)
        
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        
        output = model(x)
        
        assert output.shape == (batch_size, seq_len, config.hidden_size)
    
    def test_forward_runs_without_error(self):
        """Test that forward pass runs without errors."""
        model = GatedDeltaNet(TEST_CONFIG, layer_idx=0)
        
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, TEST_CONFIG.hidden_size)
        
        output = model(x)
        
        assert output is not None
        assert output.shape == (batch_size, seq_len, TEST_CONFIG.hidden_size)


class TestGatedDeltaNetVarlen:
    """Test GatedDeltaNetVarlen functionality."""
    
    def test_forward_shape(self):
        """Test that varlen forward pass produces correct output shape."""
        config = GatedDeltaNetConfig(
            hidden_size=64,
            num_key_heads=2,
            num_value_heads=2,
            key_head_dim=32,
            value_head_dim=32,
            conv_kernel_size=4,
        )
        model = GatedDeltaNetVarlen(config, layer_idx=0)
        
        # Three sequences with lengths 5, 10, 7
        cu_seqlens = torch.tensor([0, 5, 15, 22], dtype=torch.int32)
        total_tokens = cu_seqlens[-1].item()
        x = torch.randn(total_tokens, config.hidden_size)
        
        output = model(x, cu_seqlens=cu_seqlens)
        
        assert output.shape == (total_tokens, config.hidden_size)
    
    def test_forward_with_max_seqlen(self):
        """Test varlen forward with explicit max_seqlen."""
        model = GatedDeltaNetVarlen(TEST_CONFIG, layer_idx=0)
        
        cu_seqlens = torch.tensor([0, 8, 20], dtype=torch.int32)
        total_tokens = cu_seqlens[-1].item()
        x = torch.randn(total_tokens, TEST_CONFIG.hidden_size)
        
        output = model(x, cu_seqlens=cu_seqlens, max_seqlen=12)
        
        assert output.shape == (total_tokens, TEST_CONFIG.hidden_size)


class TestConsistency:
    """Test consistency between varlen and non-varlen versions."""
    
    @pytest.mark.parametrize("batch_size,seq_len", [(2, 16), (4, 32), (1, 64)])
    def test_same_output_equal_length(self, batch_size, seq_len):
        """Test that both versions produce identical outputs for equal-length sequences.
        
        When all sequences in a batch have the same length, varlen should produce
        the same results as the standard version.
        """
        torch.manual_seed(42)
        
        config = GatedDeltaNetConfig(
            hidden_size=64,
            num_key_heads=2,
            num_value_heads=2,
            key_head_dim=32,
            value_head_dim=32,
            conv_kernel_size=4,
            rms_norm_eps=1e-6,
            use_qk_l2norm=True,
            chunk_size=16,
        )
        
        model_base = GatedDeltaNet(config, layer_idx=0)
        model_varlen = GatedDeltaNetVarlen(config, layer_idx=0)
        
        # Copy weights to ensure identical behavior
        model_varlen.load_state_dict(model_base.state_dict())
        
        model_base.eval()
        model_varlen.eval()
        
        # Create input
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # Forward through base model
        with torch.no_grad():
            output_base = model_base(x)
        
        # Convert to packed format for varlen model
        # All sequences have same length, so cu_seqlens is evenly spaced
        cu_seqlens = torch.arange(0, batch_size + 1) * seq_len
        x_packed = x.reshape(batch_size * seq_len, config.hidden_size)
        
        # Forward through varlen model
        with torch.no_grad():
            output_varlen = model_varlen(x_packed, cu_seqlens=cu_seqlens, max_seqlen=seq_len)
        
        # Convert varlen output back to batch format
        output_varlen_batch = output_varlen.reshape(batch_size, seq_len, config.hidden_size)
        
        # Check outputs are close
        assert torch.allclose(output_base, output_varlen_batch, atol=1e-5, rtol=1e-4), \
            f"Max difference: {(output_base - output_varlen_batch).abs().max().item()}"
    
    def test_varlen_with_different_lengths(self):
        """Test varlen with sequences of different lengths.
        
        This verifies that the varlen implementation correctly handles
        padding and masking for sequences of varying lengths.
        """
        torch.manual_seed(42)
        
        config = GatedDeltaNetConfig(
            hidden_size=64,
            num_key_heads=2,
            num_value_heads=2,
            key_head_dim=32,
            value_head_dim=32,
            conv_kernel_size=4,
            rms_norm_eps=1e-6,
            use_qk_l2norm=True,
            chunk_size=16,
        )
        
        model = GatedDeltaNetVarlen(config, layer_idx=0)
        model.eval()
        
        # Three sequences with different lengths
        lengths = [10, 20, 15]
        cu_seqlens = torch.tensor([0, 10, 30, 45], dtype=torch.int32)
        total_tokens = 45
        
        x = torch.randn(total_tokens, config.hidden_size)
        
        with torch.no_grad():
            output = model(x, cu_seqlens=cu_seqlens)
        
        assert output.shape == (total_tokens, config.hidden_size)
        
        # Verify no NaN values
        assert not torch.isnan(output).any(), "Output contains NaN values"
        
        # Verify finite values
        assert torch.isfinite(output).all(), "Output contains non-finite values"
    
    def test_numerical_stability(self):
        """Test numerical stability with various input magnitudes."""
        torch.manual_seed(42)
        
        config = GatedDeltaNetConfig(
            hidden_size=32,
            num_key_heads=2,
            num_value_heads=2,
            key_head_dim=16,
            value_head_dim=16,
            conv_kernel_size=4,
        )
        
        model_base = GatedDeltaNet(config, layer_idx=0)
        model_varlen = GatedDeltaNetVarlen(config, layer_idx=0)
        model_varlen.load_state_dict(model_base.state_dict())
        model_base.eval()
        model_varlen.eval()
        
        batch_size, seq_len = 2, 16
        
        for scale in [0.01, 1.0, 10.0, 100.0]:
            x = torch.randn(batch_size, seq_len, config.hidden_size) * scale
            
            with torch.no_grad():
                output_base = model_base(x)
            
            cu_seqlens = torch.arange(0, batch_size + 1) * seq_len
            x_packed = x.reshape(batch_size * seq_len, config.hidden_size)
            
            with torch.no_grad():
                output_varlen = model_varlen(x_packed, cu_seqlens=cu_seqlens)
            
            output_varlen_batch = output_varlen.reshape(batch_size, seq_len, config.hidden_size)
            
            assert torch.allclose(output_base, output_varlen_batch, atol=1e-4, rtol=1e-3), \
                f"Scale {scale}: Max difference: {(output_base - output_varlen_batch).abs().max().item()}"
            
            # Verify no NaN or Inf
            assert torch.isfinite(output_base).all(), f"Base model output not finite at scale {scale}"
            assert torch.isfinite(output_varlen).all(), f"Varlen model output not finite at scale {scale}"


class TestParameterCounts:
    """Test parameter counts and model structure."""
    
    def test_same_parameters(self):
        """Test that both models have the same number of parameters."""
        model_base = GatedDeltaNet(TEST_CONFIG, layer_idx=0)
        model_varlen = GatedDeltaNetVarlen(TEST_CONFIG, layer_idx=0)
        
        params_base = sum(p.numel() for p in model_base.parameters())
        params_varlen = sum(p.numel() for p in model_varlen.parameters())
        
        assert params_base == params_varlen, \
            f"Parameter count mismatch: {params_base} vs {params_varlen}"
    
    def test_parameter_shapes(self):
        """Test that parameter shapes are correct."""
        model = GatedDeltaNet(TEST_CONFIG, layer_idx=0)
        
        # Check conv1d weight shape
        assert model.conv1d.weight.shape == (
            TEST_CONFIG.conv_dim, 1, TEST_CONFIG.conv_kernel_size
        ), "Conv1d weight shape incorrect"
        
        # Check output projection shape
        assert model.out_proj.weight.shape == (
            TEST_CONFIG.hidden_size,
            TEST_CONFIG.value_dim
        ), "Output projection shape incorrect"
        
        # Check input projection shapes
        expected_qkv_dim = TEST_CONFIG.key_dim * 2 + TEST_CONFIG.value_dim
        assert model.in_proj_qkv.weight.shape == (
            expected_qkv_dim,
            TEST_CONFIG.hidden_size
        ), "QKV projection shape incorrect"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_token(self):
        """Test with single token input."""
        model = GatedDeltaNet(TEST_CONFIG, layer_idx=0)
        
        x = torch.randn(2, 1, TEST_CONFIG.hidden_size)
        output = model(x)
        
        assert output.shape == (2, 1, TEST_CONFIG.hidden_size)
    
    def test_varlen_single_token_per_sequence(self):
        """Test varlen with single token per sequence."""
        model = GatedDeltaNetVarlen(TEST_CONFIG, layer_idx=0)
        
        # Three sequences, each with 1 token
        cu_seqlens = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        x = torch.randn(3, TEST_CONFIG.hidden_size)
        
        output = model(x, cu_seqlens=cu_seqlens)
        assert output.shape == (3, TEST_CONFIG.hidden_size)
    
    def test_varlen_single_sequence(self):
        """Test varlen with single sequence."""
        model = GatedDeltaNetVarlen(TEST_CONFIG, layer_idx=0)
        
        cu_seqlens = torch.tensor([0, 20], dtype=torch.int32)
        x = torch.randn(20, TEST_CONFIG.hidden_size)
        
        output = model(x, cu_seqlens=cu_seqlens)
        assert output.shape == (20, TEST_CONFIG.hidden_size)


if __name__ == "__main__":
    # Run basic tests
    print("Running basic tests...")
    
    test_basic = TestGatedDeltaNet()
    test_basic.test_forward_shape()
    print("✓ Forward shape test passed")
    
    test_basic.test_forward_runs_without_error()
    print("✓ Forward runs test passed")
    
    test_varlen = TestGatedDeltaNetVarlen()
    test_varlen.test_forward_shape()
    print("✓ Varlen forward shape test passed")
    
    test_consistency = TestConsistency()
    test_consistency.test_same_output_equal_length(2, 16)
    print("✓ Consistency test passed")
    
    test_consistency.test_varlen_with_different_lengths()
    print("✓ Different lengths test passed")
    
    print("\nAll basic tests passed!")
