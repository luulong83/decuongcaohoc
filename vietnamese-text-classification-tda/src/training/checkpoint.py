"""
Checkpoint Manager - Save/Load model states with resume capability
Supports Windows and Colab environments
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Any
import torch
import json
from datetime import datetime


class CheckpointManager:
    """
    Manages model checkpoints with resume capability
    
    Features:
    - Save/load model, optimizer, scheduler states
    - Keep best N checkpoints
    - Automatic cleanup of old checkpoints
    - Resume training from any checkpoint
    - Cross-platform support (Windows/Linux/Colab)
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        experiment_name: str,
        keep_last_n: int = 3,
        save_best: bool = True
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            experiment_name: Name of the experiment
            keep_last_n: Number of recent checkpoints to keep
            save_best: Whether to save the best checkpoint separately
        """
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        
        # Track checkpoints
        self.checkpoints = []
        self.best_score = float('-inf')
        self.best_checkpoint_path = None
        
        # Load existing checkpoints list
        self._load_checkpoint_list()
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        global_step: int,
        metrics: Dict[str, float],
        scheduler: Optional[Any] = None,
        is_best: bool = False,
        additional_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            global_step: Global training step
            metrics: Dictionary of metrics
            scheduler: Optional learning rate scheduler
            is_best: Whether this is the best checkpoint
            additional_state: Additional state to save
        
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'experiment_name': self.experiment_name
        }
        
        # Add scheduler state
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add additional state
        if additional_state is not None:
            checkpoint['additional_state'] = additional_state
        
        # Generate checkpoint filename
        checkpoint_filename = f"{self.experiment_name}_epoch{epoch:03d}_step{global_step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Update checkpoint list
        self.checkpoints.append({
            'path': str(checkpoint_path),
            'epoch': epoch,
            'global_step': global_step,
            'metrics': metrics,
            'timestamp': checkpoint['timestamp']
        })
        
        # Save best checkpoint
        if is_best and self.save_best:
            best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
            shutil.copy2(checkpoint_path, best_path)
            self.best_checkpoint_path = str(best_path)
            self.best_score = metrics.get('val_f1_macro', float('-inf'))
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        # Save checkpoint list
        self._save_checkpoint_list()
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load a checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to map tensors to
        
        Returns:
            Dictionary containing checkpoint information
        """
        # Check if checkpoint exists
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Return checkpoint info
        return {
            'epoch': checkpoint['epoch'],
            'global_step': checkpoint['global_step'],
            'metrics': checkpoint['metrics'],
            'timestamp': checkpoint['timestamp'],
            'additional_state': checkpoint.get('additional_state', {})
        }
    
    def load_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Load the best checkpoint
        
        Args:
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to map tensors to
        
        Returns:
            Dictionary containing checkpoint information
        """
        best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {best_path}")
        
        return self.load_checkpoint(
            str(best_path),
            model,
            optimizer,
            scheduler,
            device
        )
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get path to the latest checkpoint
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        if not self.checkpoints:
            return None
        
        # Sort by epoch and global_step
        sorted_checkpoints = sorted(
            self.checkpoints,
            key=lambda x: (x['epoch'], x['global_step']),
            reverse=True
        )
        
        latest = sorted_checkpoints[0]
        path = Path(latest['path'])
        
        if path.exists():
            return str(path)
        return None
    
    def list_checkpoints(self) -> list:
        """
        List all available checkpoints
        
        Returns:
            List of checkpoint dictionaries
        """
        return sorted(
            self.checkpoints,
            key=lambda x: (x['epoch'], x['global_step']),
            reverse=True
        )
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N"""
        if self.keep_last_n <= 0:
            return
        
        # Sort checkpoints by epoch and step
        sorted_checkpoints = sorted(
            self.checkpoints,
            key=lambda x: (x['epoch'], x['global_step']),
            reverse=True
        )
        
        # Keep only last N checkpoints
        if len(sorted_checkpoints) > self.keep_last_n:
            to_remove = sorted_checkpoints[self.keep_last_n:]
            
            for ckpt in to_remove:
                path = Path(ckpt['path'])
                if path.exists() and path != Path(self.best_checkpoint_path):
                    path.unlink()
                self.checkpoints.remove(ckpt)
    
    def _save_checkpoint_list(self):
        """Save the list of checkpoints to a JSON file"""
        list_path = self.checkpoint_dir / "checkpoint_list.json"
        with open(list_path, 'w', encoding='utf-8') as f:
            json.dump({
                'checkpoints': self.checkpoints,
                'best_checkpoint': self.best_checkpoint_path,
                'best_score': self.best_score
            }, f, indent=2, ensure_ascii=False)
    
    def _load_checkpoint_list(self):
        """Load the list of checkpoints from JSON file"""
        list_path = self.checkpoint_dir / "checkpoint_list.json"
        if list_path.exists():
            with open(list_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.checkpoints = data.get('checkpoints', [])
                self.best_checkpoint_path = data.get('best_checkpoint')
                self.best_score = data.get('best_score', float('-inf'))
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Get information about a checkpoint without loading it
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Dictionary containing checkpoint metadata
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load only metadata (not model weights)
        checkpoint = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=False
        )
        
        return {
            'epoch': checkpoint['epoch'],
            'global_step': checkpoint['global_step'],
            'metrics': checkpoint['metrics'],
            'timestamp': checkpoint['timestamp'],
            'experiment_name': checkpoint['experiment_name']
        }
    
    def delete_checkpoint(self, checkpoint_path: str):
        """
        Delete a specific checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint to delete
        """
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            
            # Remove from list
            self.checkpoints = [
                ckpt for ckpt in self.checkpoints
                if Path(ckpt['path']) != checkpoint_path
            ]
            self._save_checkpoint_list()
    
    def clear_all_checkpoints(self, keep_best: bool = True):
        """
        Delete all checkpoints
        
        Args:
            keep_best: Whether to keep the best checkpoint
        """
        for ckpt in self.checkpoints:
            path = Path(ckpt['path'])
            if path.exists():
                if keep_best and str(path) == self.best_checkpoint_path:
                    continue
                path.unlink()
        
        if not keep_best and self.best_checkpoint_path:
            best_path = Path(self.best_checkpoint_path)
            if best_path.exists():
                best_path.unlink()
            self.best_checkpoint_path = None
            self.best_score = float('-inf')
        
        self.checkpoints = []
        self._save_checkpoint_list()


def resume_training(
    checkpoint_manager: CheckpointManager,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    checkpoint_path: Optional[str] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Resume training from a checkpoint
    
    Args:
        checkpoint_manager: CheckpointManager instance
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Optional scheduler to load state into
        checkpoint_path: Specific checkpoint path (if None, uses latest)
        device: Device to map tensors to
    
    Returns:
        Dictionary containing resume information
    """
    # Determine which checkpoint to load
    if checkpoint_path is None:
        checkpoint_path = checkpoint_manager.get_latest_checkpoint()
        if checkpoint_path is None:
            raise ValueError("No checkpoint found to resume from")
    
    # Load checkpoint
    checkpoint_info = checkpoint_manager.load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        device
    )
    
    return checkpoint_info