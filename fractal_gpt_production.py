"""
Production-ready FractalGPT with adaptive computation and performance optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math
from collections import deque

from llm_fractal_training import FractalTransformerConfig


@dataclass
class AdaptiveComputationStats:
    """Statistics for adaptive computation"""
    layer_activation_history: List[deque]  # Rolling history per layer
    layer_skip_counts: List[int]
    layer_compute_counts: List[int]
    total_flops_saved: float
    total_flops_used: float
    
    def get_skip_rate(self, layer_idx: int) -> float:
        total = self.layer_skip_counts[layer_idx] + self.layer_compute_counts[layer_idx]
        return self.layer_skip_counts[layer_idx] / max(1, total)
    
    def get_mean_activation(self, layer_idx: int) -> float:
        if self.layer_activation_history[layer_idx]:
            return np.mean(list(self.layer_activation_history[layer_idx]))
        return 0.0


class AdaptiveFractalAttention(nn.Module):
    """Attention module with adaptive computation and FLOP counting"""
    
    def __init__(self, config: FractalTransformerConfig, level: int):
        super().__init__()
        self.config = config
        self.level = level
        self.head_dim = config.hidden_dim // config.num_heads
        self.num_heads = config.num_heads
        
        # Adaptive head allocation
        self.active_heads = self._compute_active_heads()
        
        # Projections
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Attention FLOP calculation
        # Q,K,V projections + attention computation + output projection
        self.flops_per_token = (
            3 * config.hidden_dim * config.hidden_dim +  # Q,K,V projections
            2 * config.hidden_dim * config.max_seq_length +  # Attention scores
            config.hidden_dim * config.hidden_dim  # Output projection
        )
        
    def _compute_active_heads(self) -> int:
        """All heads are active at every level"""
        return self.num_heads
        
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float]:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Compute Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Only use active heads
        query = query[:, :self.active_heads]
        key = key[:, :self.active_heads]
        value = value[:, :self.active_heads]
        
        # Compute attention scores with fractal pattern
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask FIRST (guarantees autoregressive correctness)
        causal_mask = torch.triu(
            torch.full((seq_length, seq_length), float('-inf'), device=scores.device),
            diagonal=1
        )
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # Apply fractal attention pattern (further restricts which past positions to attend)
        scores = self._apply_fractal_pattern(scores, seq_length)
        
        if attention_mask is not None:
            scores = scores + attention_mask
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.active_heads * self.head_dim)
        
        output = self.out_proj(attn_output)
        
        # Calculate actual FLOPs used (reduced by active heads ratio)
        flops_used = self.flops_per_token * seq_length * batch_size
        flops_used *= (self.active_heads / self.num_heads)
        
        return output, flops_used
        
    def _apply_fractal_pattern(self, scores: torch.Tensor, seq_length: int) -> torch.Tensor:
        """Apply level-specific attention patterns"""
        if self.level == 0:
            # Coarse: strided attention
            mask = self._create_strided_mask(seq_length, stride=4)
        elif self.level == 1:
            # Medium: mixed local/global
            mask = self._create_mixed_mask(seq_length, window=16)
        else:
            # Fine: local attention
            mask = self._create_local_mask(seq_length, window=32)
            
        mask = mask.to(scores.device)
        return scores + mask.unsqueeze(0).unsqueeze(0)
        
    def _create_strided_mask(self, seq_len: int, stride: int) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float('-inf'))
        for i in range(seq_len):
            for j in range(0, seq_len, stride):
                if j <= i:
                    mask[i, j] = 0
            mask[i, i] = 0  # Always attend to self
        return mask
        
    def _create_mixed_mask(self, seq_len: int, window: int) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float('-inf'))
        for i in range(seq_len):
            # Local window -- only past positions
            start = max(0, i - window + 1)
            end = i + 1
            mask[i, start:end] = 0
            # Global positions -- only past
            for j in range(0, seq_len, 32):
                if j <= i:
                    mask[i, j] = 0
        return mask
        
    def _create_local_mask(self, seq_len: int, window: int) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float('-inf'))
        for i in range(seq_len):
            start = max(0, i - window + 1)
            end = i + 1
            mask[i, start:end] = 0
        return mask


class AdaptiveFractalLayer(nn.Module):
    """Transformer layer with adaptive computation and dynamic thresholds"""
    
    def __init__(self, config: FractalTransformerConfig, level: int, layer_idx: int):
        super().__init__()
        self.config = config
        self.level = level
        self.layer_idx = layer_idx
        
        # Attention
        self.attention = AdaptiveFractalAttention(config, level)
        self.attention_norm = nn.LayerNorm(config.hidden_dim)
        
        # FFN with hierarchical sizing
        ffn_dim = self._compute_ffn_dim()
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, config.hidden_dim)
        )
        self.ffn_norm = nn.LayerNorm(config.hidden_dim)
        
        # Dropout for regularization (auto-disables in eval mode)
        self.attn_dropout = nn.Dropout(0.1)
        self.ffn_dropout = nn.Dropout(0.1)

        # Adaptive computation parameters
        self.base_threshold = 0.1 * (level + 1)
        self.threshold_momentum = 0.99
        self.register_buffer('running_mean_activation', torch.tensor(1.0))
        self.register_buffer('dynamic_threshold', torch.tensor(self.base_threshold))
        
        # Sparsity regularization
        self.sparsity_target = 0.3  # Target 30% skip rate
        self.sparsity_weight = 0.01
        
        # FLOP calculation for FFN
        self.ffn_flops_per_token = (
            config.hidden_dim * ffn_dim * 2 +  # Two linear layers
            ffn_dim  # GELU activation
        )
        
    def _compute_ffn_dim(self) -> int:
        base_dim = self.config.hidden_dim * 4
        # Smaller FFN for coarse levels
        reduction = self.config.fractal_levels - self.level
        return max(self.config.hidden_dim, base_dim // reduction)
        
    def update_threshold(self, activation_norm: torch.Tensor):
        """Update dynamic threshold based on activation statistics"""
        with torch.no_grad():
            # Update running mean
            self.running_mean_activation = (
                self.threshold_momentum * self.running_mean_activation +
                (1 - self.threshold_momentum) * activation_norm.mean()
            )
            
            # Set threshold to capture desired skip rate
            # Lower threshold = more skipping
            self.dynamic_threshold = self.running_mean_activation * 0.7
            
    def compute_sparsity_loss(self, skip_decision: bool) -> torch.Tensor:
        """Compute regularization loss to encourage target sparsity"""
        skip_rate = float(skip_decision)
        sparsity_loss = self.sparsity_weight * (skip_rate - self.sparsity_target) ** 2
        return torch.tensor(sparsity_loss, device=next(self.parameters()).device)
        
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                force_compute: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Compute activation norm for adaptive decision
        activation_norm = torch.norm(hidden_states, dim=-1).mean()
        self.update_threshold(activation_norm)
        
        # Adaptive computation decision
        skip_layer = False
        if self.config.adaptive_computation and not force_compute and not self.training:
            skip_layer = activation_norm < self.dynamic_threshold
            
        stats = {
            'activation_norm': activation_norm.item(),
            'threshold': self.dynamic_threshold.item(),
            'skipped': float(skip_layer),
            'flops_used': 0.0,
            'sparsity_loss': 0.0
        }
        
        if skip_layer:
            # Skip computation, return input unchanged
            stats['flops_used'] = 0
            stats['sparsity_loss'] = self.compute_sparsity_loss(True).item()
            return hidden_states, stats
            
        # Attention block
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        attn_output, attn_flops = self.attention(hidden_states, attention_mask)
        hidden_states = residual + self.attn_dropout(attn_output)

        # FFN block
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = residual + self.ffn_dropout(self.ffn(hidden_states))
        
        # Calculate total FLOPs
        ffn_flops = self.ffn_flops_per_token * seq_length * batch_size
        stats['flops_used'] = attn_flops + ffn_flops
        stats['sparsity_loss'] = self.compute_sparsity_loss(False).item()
        
        return hidden_states, stats


class ProductionFractalGPT(nn.Module):
    """Production-ready FractalGPT with all optimizations"""
    
    def __init__(self, config: FractalTransformerConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_dim)
        
        # Fractal transformer layers
        self.layers = nn.ModuleList()
        layer_idx = 0
        
        for level in range(config.fractal_levels):
            for _ in range(config.layers_per_level[level]):
                layer = AdaptiveFractalLayer(config, level, layer_idx)
                self.layers.append(layer)
                layer_idx += 1
                
        # Output
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Adaptive computation statistics
        self.compute_stats = AdaptiveComputationStats(
            layer_activation_history=[deque(maxlen=100) for _ in range(len(self.layers))],
            layer_skip_counts=[0] * len(self.layers),
            layer_compute_counts=[0] * len(self.layers),
            total_flops_saved=0.0,
            total_flops_used=0.0
        )
        
        # Parameter groups for optimizer
        self.parameter_groups = self._create_parameter_groups()
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _create_parameter_groups(self) -> List[Dict[str, Any]]:
        """Create hierarchical parameter groups"""
        groups = []
        
        # Group by fractal level
        for level in range(self.config.fractal_levels):
            level_params = []
            layer_indices = []
            
            # Find layers at this level
            layer_idx = sum(self.config.layers_per_level[:level])
            for i in range(self.config.layers_per_level[level]):
                if layer_idx + i < len(self.layers):
                    layer_indices.append(layer_idx + i)
                    level_params.extend(self.layers[layer_idx + i].parameters())
                    
            if level_params:
                groups.append({
                    'params': level_params,
                    'level': level,
                    'layer_indices': layer_indices,
                    'lr_scale': 0.5 ** level,
                    'weight_decay': 0.01 * (level + 1)
                })
                
        # Embeddings and head
        embed_params = (
            list(self.token_embedding.parameters()) +
            list(self.position_embedding.parameters()) +
            list(self.lm_head.parameters())
        )
        groups.append({
            'params': embed_params,
            'level': -1,
            'lr_scale': 1.0,
            'weight_decay': 0.01
        })
        
        return groups
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def get_activation_profile(self) -> Dict[str, List[float]]:
        """Get activation statistics for threshold tuning"""
        profile = {}
        for i, layer in enumerate(self.layers):
            profile[f'layer_{i}'] = {
                'mean_activation': self.compute_stats.get_mean_activation(i),
                'skip_rate': self.compute_stats.get_skip_rate(i),
                'threshold': layer.dynamic_threshold.item(),
                'level': i // (len(self.layers) // self.config.fractal_levels)
            }
        return profile
        
    def reset_compute_stats(self):
        """Reset computation statistics"""
        self.compute_stats = AdaptiveComputationStats(
            layer_activation_history=[deque(maxlen=100) for _ in range(len(self.layers))],
            layer_skip_counts=[0] * len(self.layers),
            layer_compute_counts=[0] * len(self.layers),
            total_flops_saved=0.0,
            total_flops_used=0.0
        )
        
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                return_stats: bool = False,
                force_compute_all: bool = False) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_length = input_ids.shape
        
        # Get embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
            
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds
        
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
            
        # Apply transformer layers
        layer_stats = []
        total_sparsity_loss = 0.0
        
        for i, layer in enumerate(self.layers):
            hidden_states, stats = layer(
                hidden_states, 
                attention_mask,
                force_compute=force_compute_all
            )
            
            # Update statistics
            self.compute_stats.layer_activation_history[i].append(stats['activation_norm'])
            if stats['skipped']:
                self.compute_stats.layer_skip_counts[i] += 1
            else:
                self.compute_stats.layer_compute_counts[i] += 1
                
            self.compute_stats.total_flops_used += stats['flops_used']
            total_sparsity_loss += stats['sparsity_loss']
            
            layer_stats.append(stats)
            
        # Final norm and output
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # Calculate total possible FLOPs and savings
        total_possible_flops = sum(
            layer.attention.flops_per_token * seq_length * batch_size +
            layer.ffn_flops_per_token * seq_length * batch_size
            for layer in self.layers
        )
        
        flops_saved = total_possible_flops - sum(s['flops_used'] for s in layer_stats)
        self.compute_stats.total_flops_saved += flops_saved
        
        output = {
            'logits': logits,
            'sparsity_loss': total_sparsity_loss
        }
        
        if return_stats:
            compute_savings = flops_saved / total_possible_flops if total_possible_flops > 0 else 0
            output.update({
                'layer_stats': layer_stats,
                'compute_savings': compute_savings,
                'total_flops_used': sum(s['flops_used'] for s in layer_stats),
                'total_flops_possible': total_possible_flops
            })
            
        return output


def create_standard_transformer(config: FractalTransformerConfig) -> nn.Module:
    """Create standard transformer baseline for comparison"""
    from transformers import GPT2Config, GPT2LMHeadModel
    
    # Convert config to standard GPT2 config
    gpt2_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.max_seq_length,
        n_embd=config.hidden_dim,
        n_layer=config.num_layers,
        n_head=config.num_heads,
        n_inner=config.hidden_dim * 4,  # Standard 4x FFN
        activation_function='gelu_new',
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0
    )
    
    return GPT2LMHeadModel(gpt2_config)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_production_model():
    """Test the production model"""
    print("Testing Production FractalGPT...")
    
    config = FractalTransformerConfig(
        vocab_size=1000,
        max_seq_length=128,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        fractal_levels=3,
        adaptive_computation=True
    )
    
    # Create models
    fractal_model = ProductionFractalGPT(config)
    standard_model = create_standard_transformer(config)
    
    print(f"\nFractalGPT parameters: {count_parameters(fractal_model):,}")
    print(f"Standard GPT parameters: {count_parameters(standard_model):,}")
    
    # Test forward pass
    batch_size = 4
    seq_length = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    # Test FractalGPT
    fractal_model.eval()
    with torch.no_grad():
        outputs = fractal_model(input_ids, return_stats=True)
        
    print(f"\nFractalGPT Results:")
    print(f"  Compute savings: {outputs['compute_savings']:.2%}")
    print(f"  FLOPs used: {outputs['total_flops_used']:,.0f}")
    print(f"  FLOPs possible: {outputs['total_flops_possible']:,.0f}")
    
    # Show per-layer statistics
    print("\nPer-layer skip rates:")
    profile = fractal_model.get_activation_profile()
    for layer_name, stats in profile.items():
        print(f"  {layer_name} (Level {stats['level']}): "
              f"skip_rate={stats['skip_rate']:.2%}, "
              f"threshold={stats['threshold']:.3f}")
    
    print("\n✓ Production model test passed!")


if __name__ == "__main__":
    test_production_model()