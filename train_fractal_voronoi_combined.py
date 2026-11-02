#!/usr/bin/env python3
"""
Combined FractalGPT architecture with Voronoi optimization
Includes anti-repetition measures and performance optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
import numpy as np
from tqdm import tqdm
import os
import time
import math
import itertools
from collections import defaultdict
from typing import Dict, Tuple, Optional

# Import FractalGPT architecture
from fractal_gpt_production import ProductionFractalGPT
from llm_fractal_training import FractalTransformerConfig

import local_logger as metrics_logger
from wikitext_data import WIKITEXT_REPO, WIKITEXT_TOKENIZER, detokenize_wikitext, wikitext_windows

TINYSTORIES_TOKENIZER = 'fractal_gpt_tokenizer_8k.json'

DATA_DESCRIPTIONS = {
    'mixed': 'TinyStories + WikiText-2',
    'tinystories': 'TinyStories only',
    'wikitext': 'WikiText-103 only',
}


class FractalVoronoiDataset(torch.utils.data.IterableDataset):
    """Dataset for training -- data is 'mixed' (TinyStories + WikiText-2), 'tinystories' or 'wikitext' (WikiText-103)"""

    def __init__(self, tokenizer, max_length=256, data='mixed'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data

        self.stories = None
        self.wiki = None
        if data != 'wikitext':
            self.stories = load_dataset('roneneldan/TinyStories', split='train', streaming=True)
        if data == 'mixed':
            self.wiki = load_dataset(WIKITEXT_REPO, 'wikitext-2-raw-v1', split='train', streaming=True)
        elif data == 'wikitext':
            self.wiki = load_dataset(WIKITEXT_REPO, 'wikitext-103-raw-v1', split='train', streaming=True)

    @staticmethod
    def _to_example(sequence):
        return {
            'input_ids': torch.tensor(sequence[:-1], dtype=torch.long),
            'labels': torch.tensor(sequence[1:], dtype=torch.long),
        }

    def _process_text(self, text, detokenize=False):
        """Tokenize and yield sequences from text"""
        if not text or len(text) < 50 or text.count(' ') < 5:
            return
        if detokenize:
            text = detokenize_wikitext(text)
        encoding = self.tokenizer.encode(text)
        tokens = encoding.ids
        for i in range(0, len(tokens) - self.max_length, self.max_length // 4):
            sequence = tokens[i:i + self.max_length]
            if len(sequence) == self.max_length:
                yield self._to_example(sequence)

    def __iter__(self):
        if self.data == 'wikitext':
            # One continuous token stream, restarted at the end of the split
            while True:
                for sequence in wikitext_windows(self.tokenizer, self.wiki, self.max_length):
                    yield self._to_example(sequence)

        stories_iter = iter(self.stories)
        wiki_iter = iter(self.wiki) if self.wiki is not None else None

        while True:
            # TinyStories
            try:
                example = next(stories_iter)
                yield from self._process_text(example.get('text', ''))
            except StopIteration:
                stories_iter = iter(self.stories)

            # WikiText (if mixed mode)
            if wiki_iter is not None:
                try:
                    example = next(wiki_iter)
                    yield from self._process_text(example.get('text', ''), detokenize=True)
                except StopIteration:
                    wiki_iter = iter(self.wiki)


def create_fractal_voronoi_model(config: dict):
    """Create FractalGPT model with proper initialization"""
    
    model_config = FractalTransformerConfig(
        vocab_size=config['vocab_size'],
        max_seq_length=config['max_seq_length'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        fractal_levels=config['fractal_levels'],
        adaptive_computation=config.get('adaptive_computation', True)
    )
    
    model = ProductionFractalGPT(model_config)
    
    # Smart initialization to avoid uniform distribution trap
    from train_fractal_voronoi_fixed_init import SmartInitializer
    SmartInitializer.initialize_model(model, config['vocab_size'])
    
    return model


def train_fractal_voronoi(config: dict):
    """Main training function combining FractalGPT + Voronoi + Anti-repetition"""
    
    print("=== FractalGPT + Voronoi Training ===")
    print("\nConfiguration:")
    for k, v in config.items():
        if k != 'device':
            print(f"  {k}: {v}")
    
    device = config['device']
    
    # Check for checkpoint resumption
    start_epoch = 0
    global_step = 0
    checkpoint_data = None
    
    if config.get('resume_checkpoint'):
        checkpoint_path = config['resume_checkpoint']
        if os.path.exists(checkpoint_path):
            print(f"\nLoading checkpoint from {checkpoint_path}")
            checkpoint_data = torch.load(checkpoint_path, weights_only=False)
            
            # Get starting epoch
            if config.get('start_epoch'):
                start_epoch = config['start_epoch']
            elif 'epoch' in checkpoint_data:
                start_epoch = checkpoint_data['epoch']
            elif 'step' in checkpoint_data and 'steps_per_epoch' in config:
                start_epoch = checkpoint_data['step'] // config['steps_per_epoch']
            
            global_step = checkpoint_data.get('step', start_epoch * config['steps_per_epoch'])
            
            print(f"Resuming from epoch {start_epoch + 1}, global step {global_step}")
            
            # Update config with checkpoint's config
            if 'config' in checkpoint_data:
                old_config = checkpoint_data['config']
                # Preserve new settings but use model architecture and tokenizer from checkpoint
                for key in ['hidden_dim', 'num_layers', 'num_heads', 'vocab_size', 'max_seq_length']:
                    if key in old_config:
                        config[key] = old_config[key]
                # Checkpoints from before --wikitext were all trained with the TinyStories tokenizer
                config['tokenizer'] = old_config.get('tokenizer', TINYSTORIES_TOKENIZER)
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found, starting fresh")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(config.get('tokenizer', TINYSTORIES_TOKENIZER))
    
    # Create model
    model = create_fractal_voronoi_model(config)
    
    # Load model weights if resuming
    if checkpoint_data and 'model_state_dict' in checkpoint_data:
        model.load_state_dict(checkpoint_data['model_state_dict'])
        print("Model weights loaded from checkpoint")
    
    model = model.to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset
    data = config.get('data', 'mixed')
    dataset = FractalVoronoiDataset(tokenizer, config['max_seq_length'], data=data)
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=0  # For streaming dataset
    )
    
    # AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01,
    )

    # Load optimizer state if resuming
    if checkpoint_data and 'optimizer_state_dict' in checkpoint_data:
        try:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            print("Optimizer state loaded from checkpoint")
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")

    # Standard cross-entropy loss
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # LR scheduler: linear warmup + cosine decay
    warmup_steps = config.get('warmup_steps', 100)
    total_steps = config['steps_per_epoch'] * config['num_epochs']

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Load scheduler state if resuming
    if checkpoint_data and 'scheduler_state_dict' in checkpoint_data:
        try:
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            print("Scheduler state loaded from checkpoint")
        except:
            print("Warning: Could not load scheduler state, using fresh scheduler")
    
    # Initialize logger
    run_name = f"fractal_voronoi_{config['hidden_dim']}d_{config['num_layers']}L"
    if data == 'wikitext':
        run_name += "_wikitext"  # different tokenizer, keep checkpoints apart
    metrics_logger.init(
        project="fractal-voronoi",
        config=config,
        name=run_name
    )
    
    # Training metrics
    metrics_history = defaultdict(list)
    best_val_loss = float('inf')
    
    # Training loop
    model.train()
    
    # Use the global_step from checkpoint if loaded, otherwise start at 0
    if not checkpoint_data:
        global_step = 0
    
    print("\nStarting training...")
    print("Features:")
    print("  - FractalGPT hierarchical architecture with causal masking")
    print("  - AdamW optimizer")
    print(f"  - Data: {DATA_DESCRIPTIONS[data]}")
    
    # Adjust for resumption
    total_epochs = config['num_epochs']
    if start_epoch > 0:
        print(f"\nResuming from epoch {start_epoch + 1}")
        if start_epoch >= total_epochs:
            print(f"Already completed {start_epoch} epochs, requested only {total_epochs}")
            return
    
    for epoch in range(start_epoch, total_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{total_epochs}")
        
        epoch_metrics = defaultdict(list)
        pbar = tqdm(desc=f"Epoch {epoch + 1}", total=config['steps_per_epoch'])
        
        for step, batch in enumerate(dataloader):
            if step >= config['steps_per_epoch']:
                break
            
            # Move to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, return_stats=True)
            logits = outputs['logits']
            
            # Compute cross-entropy loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            perplexity = math.exp(min(loss.item(), 20))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Optimizer step
            optimizer.step()
            scheduler.step()

            # Track metrics
            epoch_metrics['loss'].append(loss.item())
            epoch_metrics['perplexity'].append(perplexity)

            global_step += 1
            pbar.update(1)

            # Update progress bar
            if step % 10 == 0:
                avg_loss = np.mean(epoch_metrics['loss'][-50:])
                avg_ppl = np.mean(epoch_metrics['perplexity'][-50:])

                pbar.set_postfix({
                    'loss': f"{avg_loss:.3f}",
                    'ppl': f"{avg_ppl:.1f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })

            # Log metrics
            if global_step % 100 == 0:
                log_metrics = {
                    f"train/{k}": np.mean(v[-100:])
                    for k, v in epoch_metrics.items()
                }
                log_metrics['train/lr'] = optimizer.param_groups[0]['lr']
                log_metrics['train/epoch'] = epoch
                log_metrics['train/step'] = global_step

                metrics_logger.log(log_metrics)
            
            # Periodic evaluation
            if global_step % 1000 == 0 and global_step > 0:
                print(f"\nEvaluating at step {global_step}...")
                val_loss = evaluate_model(model, tokenizer, device, data, config['max_seq_length'])
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = f"{run_name}_best.pt"
                    save_checkpoint(model, optimizer, scheduler, config, global_step, epoch, save_path)
                    print(f"  New best model saved to {save_path}")
                
                model.train()
        
        pbar.close()
        
        # Epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Avg loss: {np.mean(epoch_metrics['loss']):.3f}")
        print(f"  Avg perplexity: {np.mean(epoch_metrics['perplexity']):.1f}")

        # Test generation
        print("\nSample generation:")
        test_generation(model, tokenizer, device, data=data)
    
    # Final save
    final_path = f"{run_name}_final.pt"
    save_checkpoint(model, optimizer, scheduler, config, global_step, total_epochs - 1, final_path)
    print(f"\n✓ Training complete! Final model saved to {final_path}")
    
    metrics_logger.finish()


def validation_sequences(tokenizer, data='mixed', max_length=256, count=50):
    """Small validation set as lists of token ids (TinyStories unless training on WikiText only)"""
    if data == 'wikitext':
        val_data = load_dataset(WIKITEXT_REPO, 'wikitext-103-raw-v1', split='validation', streaming=True)
        return list(itertools.islice(wikitext_windows(tokenizer, val_data, max_length), count))

    sequences = []
    val_data = load_dataset('roneneldan/TinyStories', split='validation', streaming=True)
    for i, example in enumerate(val_data):
        if i >= count:  # Quick evaluation
            break

        text = example.get('text', '')
        if text and len(text) > 100:
            ids = tokenizer.encode(text[:500]).ids  # Limit length
            if len(ids) > 10:
                sequences.append(ids)
    return sequences


def evaluate_model(model, tokenizer, device, data='mixed', max_length=256):
    """Quick evaluation on validation data"""
    model.eval()

    # Simple validation loss calculation
    val_losses = []

    with torch.no_grad():
        for ids in validation_sequences(tokenizer, data, max_length):
            tokens = torch.tensor(ids).unsqueeze(0).to(device)
            input_ids = tokens[:, :-1]
            labels = tokens[:, 1:]

            outputs = model(input_ids)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            val_losses.append(loss.item())
    
    avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
    print(f"  Validation loss: {avg_val_loss:.3f}")
    print(f"  Validation perplexity: {np.exp(avg_val_loss):.1f}")
    
    return avg_val_loss


def test_generation(model, tokenizer, device, num_samples=3, data='mixed'):
    """Test text generation quality"""
    model.eval()

    if data == 'wikitext':
        prompts = ["The city of", "In the early 20th century", "The album was"]
    else:
        prompts = ["Once upon a time", "The little girl", "In a magical forest"]
    
    for prompt in prompts[:num_samples]:
        encoding = tokenizer.encode(prompt)
        input_ids = torch.tensor(encoding.ids).unsqueeze(0).to(device)
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(30):
                outputs = model(generated)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                # Apply temperature and penalties
                next_logits = logits[0, -1, :] / 0.9
                
                # Light repetition penalty
                for i in range(generated.size(1)):
                    token = generated[0, i].item()
                    penalty = 1.1 - (i / generated.size(1)) * 0.1
                    next_logits[token] /= penalty
                
                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        result = tokenizer.decode(generated[0].cpu().tolist())
        print(f"  '{prompt}' → {result}")
    
    model.train()


def save_checkpoint(model, optimizer, scheduler, config, step, epoch, path):
    """Save training checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
        'step': step,
        'epoch': epoch,
    }
    torch.save(checkpoint, path)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train FractalGPT with Voronoi optimization')
    parser.add_argument('--tiny', action='store_true', help='Tiny model (2M params)')
    parser.add_argument('--small', action='store_true', help='Small model (8M params)')
    parser.add_argument('--medium', action='store_true', help='Medium model (26M params, default)')
    parser.add_argument('--steps', type=int, help='Override steps per epoch')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint file')
    parser.add_argument('--start-epoch', type=int, help='Starting epoch (for resuming)')
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument('--tinystories', action='store_true', help='Train on TinyStories only (easier, lower loss)')
    data_group.add_argument('--wikitext', action='store_true', help='Train on WikiText-103 only (uses the WikiText tokenizer)')
    
    args = parser.parse_args()
    
    # Base configuration
    config = {
        'vocab_size': 8000,
        'max_seq_length': 256,
        'fractal_levels': 3,
        'adaptive_computation': False,
        'dropout': 0.1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    # Model size presets
    if args.tiny:
        config.update({
            'hidden_dim': 128,
            'num_layers': 4,
            'num_heads': 2,
            'batch_size': 32,
            'learning_rate': 1e-3,  # Increased for faster convergence
            'num_epochs': 5,
            'steps_per_epoch': 1000,
            'warmup_steps': 100,
        })
        print("Using TINY model configuration (~2M params)")
    elif args.small:
        config.update({
            'hidden_dim': 256,
            'num_layers': 6,
            'num_heads': 4,
            'batch_size': 24,
            'learning_rate': 3e-4,
            'num_epochs': 8,
            'steps_per_epoch': 1500,
        })
        print("Using SMALL model configuration (~8M params)")
    else:  # medium (default)
        config.update({
            'hidden_dim': 512,
            'num_layers': 8,
            'num_heads': 8,
            'batch_size': 32,
            'learning_rate': 3e-4,
            'num_epochs': 20,
            'steps_per_epoch': 5000,
        })
        if not args.medium:
            print("Using MEDIUM model configuration (~26M params, default)")
    
    # Override with command line args
    if args.steps:
        config['steps_per_epoch'] = args.steps
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr
    
    # Data settings
    if args.tinystories:
        config['data'] = 'tinystories'
    elif args.wikitext:
        config['data'] = 'wikitext'
    else:
        config['data'] = 'mixed'
    config['tokenizer'] = WIKITEXT_TOKENIZER if config['data'] == 'wikitext' else TINYSTORIES_TOKENIZER
    print(f"Training on {DATA_DESCRIPTIONS[config['data']]}")

    # Resume settings
    if args.resume:
        config['resume_checkpoint'] = args.resume
        if args.start_epoch:
            config['start_epoch'] = args.start_epoch
    
    # Adjust for CPU if needed
    if config['device'] == 'cpu':
        config['batch_size'] = max(4, config['batch_size'] // 4)
        # Only cap steps if user didn't explicitly set them
        if not args.steps:
            config['steps_per_epoch'] = min(500, config['steps_per_epoch'])
        print("Adjusted configuration for CPU training")
    
    train_fractal_voronoi(config)


if __name__ == "__main__":
    main()