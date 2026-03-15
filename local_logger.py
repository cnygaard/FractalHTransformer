"""
Local logging alternative to Weights & Biases (wandb)
Provides similar API but stores everything locally without cloud telemetry
"""

import json
import os
import time
import datetime
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle


class LocalLogger:
    """
    Local alternative to wandb that provides similar functionality
    without sending any data to the cloud.
    """
    
    def __init__(self):
        self.run = None
        self.config = {}
        self.metrics_history = []
        self.images = []
        self.run_dir = None
        self.start_time = None
        
    def init(self, project: str, name: Optional[str] = None, 
             config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize a new run"""
        # Create run directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = name or f"run_{timestamp}"
        
        base_dir = Path("runs") / project
        base_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_dir = base_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.config = config or {}
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)
            
        # Initialize run info
        self.run = {
            "id": run_name,
            "name": run_name,
            "project": project,
            "dir": str(self.run_dir)
        }
        
        self.start_time = time.time()
        
        # Create subdirectories
        (self.run_dir / "metrics").mkdir(exist_ok=True)
        (self.run_dir / "images").mkdir(exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        
        print(f"LocalLogger: Started run {run_name} in {self.run_dir}")
        return self.run
        
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics for the current step"""
        if self.run is None:
            return
            
        # Add timestamp and step
        log_entry = {
            "step": step or len(self.metrics_history),
            "timestamp": time.time() - self.start_time,
            **metrics
        }
        
        self.metrics_history.append(log_entry)
        
        # Save metrics incrementally
        if len(self.metrics_history) % 100 == 0:
            self._save_metrics()
            
    def log_image(self, key: str, image: Any, step: Optional[int] = None):
        """Log an image (matplotlib figure or numpy array)"""
        if self.run is None:
            return
            
        step = step or len(self.metrics_history)
        
        # Save image
        if isinstance(image, plt.Figure):
            image_path = self.run_dir / "images" / f"{key}_step{step}.png"
            image.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close(image)
        elif isinstance(image, np.ndarray):
            image_path = self.run_dir / "images" / f"{key}_step{step}.npy"
            np.save(image_path, image)
        else:
            # Try to convert to figure
            fig = plt.figure()
            plt.imshow(image)
            plt.title(key)
            image_path = self.run_dir / "images" / f"{key}_step{step}.png"
            fig.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        self.images.append({
            "key": key,
            "step": step,
            "path": str(image_path)
        })
        
    def finish(self):
        """Finish the current run and save all data"""
        if self.run is None:
            return
            
        # Save final metrics
        self._save_metrics()
        
        # Save summary
        summary = {
            "run_id": self.run["id"],
            "duration": time.time() - self.start_time,
            "total_steps": len(self.metrics_history),
            "metrics": self._compute_summary_stats(),
            "images": self.images
        }
        
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        print(f"LocalLogger: Finished run. Data saved to {self.run_dir}")
        
        # Generate plots
        self._generate_plots()
        
        self.run = None
        
    def _save_metrics(self):
        """Save metrics history to disk"""
        if not self.metrics_history:
            return
            
        # Save as JSON for easy reading
        with open(self.run_dir / "metrics" / "history.json", "w") as f:
            json.dump(self.metrics_history, f, indent=2)
            
        # Save as pickle for faster loading
        with open(self.run_dir / "metrics" / "history.pkl", "wb") as f:
            pickle.dump(self.metrics_history, f)
            
    def _compute_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics for all metrics"""
        if not self.metrics_history:
            return {}
            
        # Extract metric names (excluding step and timestamp)
        metric_names = set()
        for entry in self.metrics_history:
            metric_names.update(k for k in entry.keys() 
                              if k not in ['step', 'timestamp'])
        
        summary = {}
        for metric in metric_names:
            values = [entry[metric] for entry in self.metrics_history 
                     if metric in entry]
            if values:
                summary[metric] = {
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values)),
                    'final': float(values[-1])
                }
                
        return summary
        
    def _generate_plots(self):
        """Generate plots for all logged metrics"""
        if not self.metrics_history:
            return
            
        # Extract metrics
        metrics_data = {}
        steps = []
        
        for entry in self.metrics_history:
            steps.append(entry['step'])
            for key, value in entry.items():
                if key not in ['step', 'timestamp']:
                    if key not in metrics_data:
                        metrics_data[key] = []
                    metrics_data[key].append(value)
        
        # Create plots
        for metric_name, values in metrics_data.items():
            # Ensure we have matching lengths
            metric_steps = steps[:len(values)]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(metric_steps, values, linewidth=2)
            ax.set_xlabel('Step')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} over time')
            ax.grid(True, alpha=0.3)
            
            # Handle nested metric names (e.g., "train/loss")
            metric_parts = metric_name.split('/')
            if len(metric_parts) > 1:
                # Create subdirectory if needed
                subdir = self.run_dir / "metrics" / '/'.join(metric_parts[:-1])
                subdir.mkdir(parents=True, exist_ok=True)
                plot_path = self.run_dir / "metrics" / f"{metric_name.replace('/', '_')}.png"
            else:
                plot_path = self.run_dir / "metrics" / f"{metric_name}.png"
            
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)


# Global instance to mimic wandb API
_logger = LocalLogger()


# Wandb-compatible API
def init(project: str, name: Optional[str] = None, 
         config: Optional[Dict[str, Any]] = None, **kwargs):
    """Initialize a new run"""
    return _logger.init(project, name, config, **kwargs)


def log(metrics: Dict[str, Any], step: Optional[int] = None):
    """Log metrics"""
    _logger.log(metrics, step)


def finish():
    """Finish the current run"""
    _logger.finish()


def log_image(key: str, image: Any, step: Optional[int] = None):
    """Log an image"""
    _logger.log_image(key, image, step)


# Convenience access to run info
class Run:
    @property
    def id(self):
        return _logger.run["id"] if _logger.run else None
        
    @property
    def name(self):
        return _logger.run["name"] if _logger.run else None
        
    @property
    def dir(self):
        return str(_logger.run["dir"]) if _logger.run else None


run = Run()


# Additional utilities
def load_run(run_dir: str) -> Dict[str, Any]:
    """Load data from a previous run"""
    run_path = Path(run_dir)
    
    # Load config
    with open(run_path / "config.json", "r") as f:
        config = json.load(f)
        
    # Load metrics
    metrics_file = run_path / "metrics" / "history.pkl"
    if metrics_file.exists():
        with open(metrics_file, "rb") as f:
            metrics = pickle.load(f)
    else:
        with open(run_path / "metrics" / "history.json", "r") as f:
            metrics = json.load(f)
            
    # Load summary
    with open(run_path / "summary.json", "r") as f:
        summary = json.load(f)
        
    return {
        "config": config,
        "metrics": metrics,
        "summary": summary
    }


def compare_runs(run_dirs: List[str], metric: str) -> plt.Figure:
    """Compare a specific metric across multiple runs"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for run_dir in run_dirs:
        data = load_run(run_dir)
        metrics = data["metrics"]
        
        steps = [entry["step"] for entry in metrics if metric in entry]
        values = [entry[metric] for entry in metrics if metric in entry]
        
        run_name = Path(run_dir).name
        ax.plot(steps, values, label=run_name, linewidth=2)
        
    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.set_title(f"Comparison: {metric}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig