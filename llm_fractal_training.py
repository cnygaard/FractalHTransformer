"""
Large Language Model Training with Fractal Ellipsoid Decomposition
"""

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, NamedTuple, Any
from dataclasses import dataclass
import math

from ellipsoid_core import Ellipsoid, create_ellipsoid, decompose_ellipsoid
from fractal_tree import FractalNode, FractalTree
from adaptive_moments import HierarchicalMoments


class FractalTransformerConfig:
    """Configuration for fractal transformer architecture"""
    
    def __init__(self,
                 vocab_size: int = 50257,
                 max_seq_length: int = 2048,
                 hidden_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 fractal_levels: int = 3,
                 hierarchical_partitioning: bool = True,
                 adaptive_computation: bool = True):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.fractal_levels = fractal_levels
        self.hierarchical_partitioning = hierarchical_partitioning
        self.adaptive_computation = adaptive_computation
        
        # Fractal decomposition of layers
        self.layers_per_level = self._compute_layer_distribution()
        
    def _compute_layer_distribution(self) -> List[int]:
        """Distribute layers across fractal levels"""
        base_layers = self.num_layers // self.fractal_levels
        remainder = self.num_layers % self.fractal_levels
        
        distribution = [base_layers] * self.fractal_levels
        for i in range(remainder):
            distribution[i] += 1
            
        return distribution


class FractalAttention(nn.Module):
    """Multi-head attention with fractal pattern constraints"""
    
    def __init__(self, config: FractalTransformerConfig, level: int):
        super().__init__()
        self.config = config
        self.level = level  # Fractal level (0 = coarse, higher = fine)
        
        self.head_dim = config.hidden_dim // config.num_heads
        self.num_heads = config.num_heads
        
        # Hierarchical head allocation
        self.active_heads = self._compute_active_heads()
        
        # Projections
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Ellipsoid for attention pattern constraints
        self.attention_ellipsoid = None
        self._init_attention_ellipsoid()
        
    def _compute_active_heads(self) -> int:
        """Compute number of active heads based on fractal level"""
        # Higher levels use more heads (finer attention)
        min_heads = max(1, self.num_heads // self.config.fractal_levels)
        return min_heads * (self.level + 1)
        
    def _init_attention_ellipsoid(self):
        """Initialize ellipsoid for attention pattern constraints"""
        # Create ellipsoid in attention space
        att_dim = self.config.max_seq_length
        scale = 1.0 / (self.level + 1)  # Tighter constraints at higher levels
        self.attention_ellipsoid = create_ellipsoid(att_dim, scale=scale)
        
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Compute Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Only use active heads based on fractal level
        query = query[:, :self.active_heads]
        key = key[:, :self.active_heads]
        value = value[:, :self.active_heads]
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply fractal attention pattern
        scores = self._apply_fractal_pattern(scores, seq_length)
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores + attention_mask
            
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.active_heads * self.head_dim)
        
        # Pad if using fewer heads
        if self.active_heads < self.num_heads:
            padding = torch.zeros(
                batch_size, seq_length,
                (self.num_heads - self.active_heads) * self.head_dim,
                device=attn_output.device
            )
            attn_output = torch.cat([attn_output, padding], dim=-1)
            
        output = self.out_proj(attn_output)
        
        return output
        
    def _apply_fractal_pattern(self, scores: torch.Tensor, seq_length: int) -> torch.Tensor:
        """Apply fractal constraints to attention pattern"""
        batch_size, num_heads, _, _ = scores.shape
        
        # Create fractal mask based on level
        if self.level == 0:
            # Coarse level: attend to distant tokens
            mask = self._create_strided_mask(seq_length, stride=4)
        elif self.level == 1:
            # Medium level: local + some distant
            mask = self._create_mixed_mask(seq_length)
        else:
            # Fine level: mostly local attention
            mask = self._create_local_mask(seq_length, window_size=32)
            
        # Apply mask
        mask = mask.to(scores.device)
        scores = scores + mask
        
        return scores
        
    def _create_strided_mask(self, seq_length: int, stride: int) -> torch.Tensor:
        """Create strided attention mask"""
        mask = torch.full((seq_length, seq_length), float('-inf'))
        for i in range(seq_length):
            for j in range(0, seq_length, stride):
                mask[i, j] = 0
        return mask
        
    def _create_mixed_mask(self, seq_length: int) -> torch.Tensor:
        """Create mixed local/global attention mask"""
        mask = torch.full((seq_length, seq_length), float('-inf'))
        window = 16
        
        for i in range(seq_length):
            # Local window
            start = max(0, i - window)
            end = min(seq_length, i + window + 1)
            mask[i, start:end] = 0
            
            # Some global positions
            global_positions = list(range(0, seq_length, 32))
            for pos in global_positions:
                if pos < seq_length:
                    mask[i, pos] = 0
                    
        return mask
        
    def _create_local_mask(self, seq_length: int, window_size: int) -> torch.Tensor:
        """Create local attention mask"""
        mask = torch.full((seq_length, seq_length), float('-inf'))
        
        for i in range(seq_length):
            start = max(0, i - window_size // 2)
            end = min(seq_length, i + window_size // 2 + 1)
            mask[i, start:end] = 0
            
        return mask


class FractalTransformerLayer(nn.Module):
    """Transformer layer with fractal decomposition"""
    
    def __init__(self, config: FractalTransformerConfig, level: int, layer_idx: int):
        super().__init__()
        self.config = config
        self.level = level
        self.layer_idx = layer_idx
        
        # Attention with fractal patterns
        self.attention = FractalAttention(config, level)
        self.attention_norm = nn.LayerNorm(config.hidden_dim)
        
        # FFN with hierarchical sizing
        ffn_dim = self._compute_ffn_dim()
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, config.hidden_dim)
        )
        self.ffn_norm = nn.LayerNorm(config.hidden_dim)
        
        # Adaptive computation threshold
        self.computation_threshold = 0.1 * (level + 1)
        
    def _compute_ffn_dim(self) -> int:
        """Compute FFN dimension based on fractal level"""
        base_dim = self.config.hidden_dim * 4
        # Smaller FFN for coarse levels
        return base_dim // (self.config.fractal_levels - self.level)
        
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                use_adaptive: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Adaptive computation: skip if activation is low
        if use_adaptive and self.config.adaptive_computation:
            activation_norm = torch.norm(hidden_states, dim=-1).mean()
            if activation_norm < self.computation_threshold:
                return hidden_states, {'skipped': 1.0}
                
        # Attention
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        attn_output = self.attention(hidden_states, attention_mask)
        hidden_states = residual + attn_output
        
        # FFN
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = residual + ffn_output
        
        stats = {'skipped': 0.0, 'activation_norm': torch.norm(hidden_states, dim=-1).mean().item()}
        
        return hidden_states, stats


class FractalGPT(nn.Module):
    """GPT-style model with fractal hierarchical decomposition"""
    
    def __init__(self, config: FractalTransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_dim)
        
        # Fractal transformer layers
        self.layers = nn.ModuleList()
        layer_idx = 0
        
        for level in range(config.fractal_levels):
            for _ in range(config.layers_per_level[level]):
                layer = FractalTransformerLayer(config, level, layer_idx)
                self.layers.append(layer)
                layer_idx += 1
                
        # Output
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Hierarchical parameter groups for optimizer
        self.parameter_groups = self._create_parameter_groups()
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _create_parameter_groups(self) -> List[Dict[str, Any]]:
        """Create hierarchical parameter groups for fractal optimization"""
        groups = []
        
        # Group by fractal level
        for level in range(self.config.fractal_levels):
            level_params = []
            level_layers = []
            
            layer_idx = sum(self.config.layers_per_level[:level])
            for i in range(self.config.layers_per_level[level]):
                if layer_idx + i < len(self.layers):
                    level_layers.append(self.layers[layer_idx + i])
                    
            for layer in level_layers:
                level_params.extend(layer.parameters())
                
            if level_params:
                groups.append({
                    'params': level_params,
                    'level': level,
                    'lr_scale': 0.5 ** level,  # Lower LR for finer levels
                    'weight_decay': 0.1 * (level + 1)
                })
                
        # Embeddings and head as separate group
        embed_params = list(self.token_embedding.parameters()) + \
                      list(self.position_embedding.parameters()) + \
                      list(self.lm_head.parameters())
        groups.append({
            'params': embed_params,
            'level': -1,  # Special level for embeddings
            'lr_scale': 1.0,
            'weight_decay': 0.01
        })
        
        return groups
        
    def _init_weights(self, module):
        """Initialize weights with level-aware scaling"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                return_stats: bool = False) -> Dict[str, torch.Tensor]:
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
            
        # Apply transformer layers with stats collection
        layer_stats = []
        for layer in self.layers:
            hidden_states, stats = layer(hidden_states, attention_mask)
            layer_stats.append(stats)
            
        # Final norm and output
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        output = {'logits': logits}
        
        if return_stats:
            output['layer_stats'] = layer_stats
            output['compute_savings'] = sum(s.get('skipped', 0) for s in layer_stats) / len(layer_stats)
            
        return output


class FractalLLMTrainer:
    """Trainer for LLMs with fractal optimization"""
    
    def __init__(self, model: FractalGPT, learning_rate: float = 1e-4):
        self.model = model
        self.base_lr = learning_rate
        
        # Create fractal optimizer with hierarchical groups
        self.optimizer = self._create_fractal_optimizer()
        
        # Ellipsoid constraints per parameter group
        self.ellipsoids = self._init_ellipsoids()
        
        # Adaptive computation scheduler
        self.computation_budget = 1.0
        
    def _create_fractal_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with fractal parameter groups"""
        # Use different learning rates per level
        param_groups = []
        
        for group in self.model.parameter_groups:
            param_groups.append({
                'params': group['params'],
                'lr': self.base_lr * group['lr_scale'],
                'weight_decay': group['weight_decay']
            })
            
        return torch.optim.AdamW(param_groups)
        
    def _init_ellipsoids(self) -> Dict[int, Any]:
        """Initialize ellipsoid constraints per level"""
        ellipsoids = {}
        
        for group in self.model.parameter_groups:
            level = group['level']
            param_count = sum(p.numel() for p in group['params'])
            
            # Create ellipsoid with level-specific scale
            scale = 10.0 / (level + 2)  # Tighter constraints for finer levels
            
            # Use JAX for ellipsoid operations
            ellipsoids[level] = create_ellipsoid(
                min(param_count, 1000),  # Cap dimension for efficiency
                scale=scale
            )
            
        return ellipsoids
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with fractal optimization"""
        self.model.train()
        
        # Forward pass with adaptive computation
        outputs = self.model(
            batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            return_stats=True
        )
        
        # Compute loss
        logits = outputs['logits']
        labels = batch['labels']
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        # Backward pass
        loss.backward()
        
        # Apply fractal constraints before optimizer step
        self._apply_fractal_constraints()
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Update computation budget based on performance
        self._update_computation_budget(outputs.get('compute_savings', 0))
        
        return {
            'loss': loss.item(),
            'compute_savings': outputs.get('compute_savings', 0),
            'perplexity': torch.exp(loss).item()
        }
        
    def _apply_fractal_constraints(self):
        """Apply ellipsoid constraints to gradients"""
        for group in self.model.parameter_groups:
            level = group['level']
            if level not in self.ellipsoids:
                continue
                
            ellipsoid = self.ellipsoids[level]
            
            # Collect and flatten gradients
            grads = []
            shapes = []
            
            for param in group['params']:
                if param.grad is not None:
                    grads.append(param.grad.data.flatten())
                    shapes.append(param.grad.shape)
                    
            if not grads:
                continue
                
            # Concatenate gradients
            flat_grad = torch.cat(grads)
            
            # Project onto ellipsoid if needed (simplified)
            grad_norm = torch.norm(flat_grad)
            max_norm = float(ellipsoid.eigenvalues.max()) * 2  # Approximate constraint
            
            if grad_norm > max_norm:
                scale = max_norm / grad_norm
                flat_grad *= scale
                
                # Reshape back
                offset = 0
                grad_idx = 0
                for param in group['params']:
                    if param.grad is not None:
                        numel = param.grad.numel()
                        param.grad.data = flat_grad[offset:offset+numel].view(shapes[grad_idx])
                        offset += numel
                        grad_idx += 1
                        
    def _update_computation_budget(self, savings: float):
        """Update adaptive computation budget"""
        # Increase budget if we're saving too much compute
        if savings > 0.5:
            self.computation_budget *= 0.95
        elif savings < 0.1:
            self.computation_budget *= 1.05
            
        self.computation_budget = np.clip(self.computation_budget, 0.5, 1.5)


def test_fractal_llm():
    """Test fractal LLM implementation"""
    print("Testing Fractal LLM Training")
    print("=" * 50)
    
    # Small config for testing
    config = FractalTransformerConfig(
        vocab_size=1000,
        max_seq_length=128,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        fractal_levels=3
    )
    
    print(f"\nModel configuration:")
    print(f"- Layers: {config.num_layers} ({config.layers_per_level} per level)")
    print(f"- Hidden dim: {config.hidden_dim}")
    print(f"- Fractal levels: {config.fractal_levels}")
    
    # Create model
    model = FractalGPT(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"- Total parameters: {total_params:,}")
    
    # Create trainer
    trainer = FractalLLMTrainer(model)
    
    # Test forward pass
    batch_size = 4
    seq_length = 64
    
    batch = {
        'input_ids': torch.randint(0, config.vocab_size, (batch_size, seq_length)),
        'labels': torch.randint(0, config.vocab_size, (batch_size, seq_length))
    }
    
    # Training step
    print("\nRunning training steps...")
    for step in range(5):
        metrics = trainer.train_step(batch)
        print(f"Step {step + 1}: Loss = {metrics['loss']:.4f}, "
              f"Compute savings = {metrics['compute_savings']:.2%}, "
              f"Perplexity = {metrics['perplexity']:.2f}")
    
    # Test hierarchical parameter groups
    print("\nParameter groups:")
    for i, group in enumerate(model.parameter_groups):
        param_count = sum(p.numel() for p in group['params'])
        print(f"  Group {i} (level {group['level']}): "
              f"{param_count:,} params, "
              f"lr_scale = {group['lr_scale']}, "
              f"weight_decay = {group['weight_decay']}")
    
    print("\nFractal LLM test completed!")


if __name__ == "__main__":
    test_fractal_llm()