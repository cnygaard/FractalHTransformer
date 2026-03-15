#!/usr/bin/env python3
"""
Fixed initialization for FractalGPT to avoid uniform distribution trap
Key changes:
1. Better weight initialization
2. Label smoothing to provide gradient signal
3. Smaller vocabulary or subword warmup
4. Temperature-based warmup
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class SmartInitializer:
    """Smart initialization to break symmetry and avoid uniform trap"""
    
    @staticmethod
    def initialize_model(model, vocab_size):
        """Initialize model weights to avoid uniform predictions"""
        
        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue
                
            # Output projection (lm_head) - most critical
            if 'lm_head' in name or 'output_projection' in name or param.shape[-1] == vocab_size:
                # Initialize with smaller variance to create initial preferences
                nn.init.normal_(param, mean=0.0, std=0.02)
                # Add structured bias to prefer common tokens
                with torch.no_grad():
                    # Make first 1000 tokens slightly more likely (common words)
                    if param.shape[-1] == vocab_size:
                        param[:, :1000] += 0.1
                        
            # Embeddings - create diverse representations
            elif 'embed' in name:
                nn.init.normal_(param, mean=0.0, std=0.1)
                
            # Attention layers - smaller init for stability
            elif 'attn' in name or 'attention' in name:
                nn.init.xavier_uniform_(param, gain=0.5)
                
            # Other layers
            else:
                nn.init.xavier_uniform_(param, gain=1.0)
        
        print("Applied smart initialization to break symmetry")


class WarmupCrossEntropyLoss(nn.Module):
    """Cross entropy with warmup tricks to escape uniform distribution"""
    
    def __init__(self, vocab_size: int, warmup_steps: int = 500):
        super().__init__()
        self.vocab_size = vocab_size
        self.warmup_steps = warmup_steps
        self.step = 0
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len, vocab_size = logits.shape
        
        # Temperature scaling during warmup (makes distribution peakier)
        if self.step < self.warmup_steps:
            temperature = 0.1 + 0.9 * (self.step / self.warmup_steps)
            logits = logits / temperature
        
        # Label smoothing to provide gradient everywhere
        smoothing = 0.1 if self.step < self.warmup_steps else 0.05
        
        # Create smoothed targets
        with torch.no_grad():
            targets = torch.zeros_like(logits).scatter_(-1, labels.unsqueeze(-1), 1.0)
            targets = targets * (1 - smoothing) + smoothing / vocab_size
        
        # Compute loss with smoothed targets
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(targets * log_probs).sum(dim=-1).mean()
        
        # Regular CE for comparison
        ce_loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
        
        # Calculate metrics
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
            max_prob = probs.max(dim=-1)[0].mean()
            
        self.step += 1
        
        return loss, {
            'loss': loss.item(),
            'ce_loss': ce_loss.item(),
            'entropy': entropy.item(),
            'max_prob': max_prob.item(),
            'temperature': temperature if self.step <= self.warmup_steps else 1.0,
            'perplexity': torch.exp(ce_loss).item()
        }


class GradientAmplifier:
    """Amplify gradients during early training to escape local minima"""
    
    def __init__(self, model, warmup_steps=200):
        self.model = model
        self.warmup_steps = warmup_steps
        self.step = 0
        
        # Register gradient hooks
        self.hooks = []
        for param in model.parameters():
            if param.requires_grad:
                hook = param.register_hook(self._gradient_hook)
                self.hooks.append(hook)
    
    def _gradient_hook(self, grad):
        """Amplify gradients during warmup"""
        if self.step < self.warmup_steps and grad is not None:
            # Amplify by up to 2x during warmup
            amplification = 1.0 + (1.0 - self.step / self.warmup_steps)
            return grad * amplification
        return grad
    
    def step_update(self):
        self.step += 1
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def create_focused_vocabulary_mask(vocab_size, focus_size=2000):
    """Create mask to focus on subset of vocabulary initially"""
    mask = torch.zeros(vocab_size)
    mask[:focus_size] = 1.0
    # Smooth transition
    transition_size = 500
    for i in range(transition_size):
        mask[focus_size + i] = 1.0 - (i / transition_size)
    return mask


def apply_vocabulary_focusing(logits, mask, step, warmup_steps=1000):
    """Gradually expand from focused vocabulary to full vocabulary"""
    if step < warmup_steps:
        # Interpolate between focused and full vocabulary
        focus_strength = 1.0 - (step / warmup_steps)
        # Apply mask with decreasing strength
        logits = logits - (1 - mask) * 10.0 * focus_strength
    return logits