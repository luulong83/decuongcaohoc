"""
Custom Logger for Vietnamese Text Classification with TDA
Supports console, file, and TensorBoard logging with error tracking
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from loguru import logger
from torch.utils.tensorboard import SummaryWriter
import colorama

# Initialize colorama for Windows
colorama.init(autoreset=True)


class ExperimentLogger:
    """
    Comprehensive logging system with:
    - Console output (colored)
    - File logging (with rotation)
    - TensorBoard integration
    - Error tracking
    - Resume support
    """
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "experiments/logs",
        tensorboard_dir: str = "experiments/logs/tensorboard",
        level: str = "INFO",
        console: bool = True,
        file: bool = True,
        tensorboard: bool = True
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.tensorboard_dir = Path(tensorboard_dir)
        self.level = level
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup loguru
        logger.remove()  # Remove default handler
        
        # Console handler
        if console:
            logger.add(
                sys.stderr,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                       "<level>{level: <8}</level> | "
                       "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                       "<level>{message}</level>",
                level=level,
                colorize=True
            )
        
        # File handler
        if file:
            self.log_file = self.log_dir / f"{experiment_name}_{self.timestamp}.log"
            logger.add(
                self.log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                level=level,
                rotation="100 MB",  # Rotate when file reaches 100MB
                retention="30 days",  # Keep logs for 30 days
                compression="zip"
            )
            
            # Separate error log
            self.error_log = self.log_dir / "errors" / f"{experiment_name}_errors.log"
            self.error_log.parent.mkdir(exist_ok=True)
            logger.add(
                self.error_log,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                level="ERROR",
                rotation="50 MB",
                retention="60 days"
            )
        
        # TensorBoard writer
        self.tensorboard_enabled = tensorboard
        if tensorboard:
            tb_path = self.tensorboard_dir / experiment_name / self.timestamp
            self.writer = SummaryWriter(log_dir=str(tb_path))
        else:
            self.writer = None
        
        # Tracking
        self.global_step = 0
        self.epoch = 0
        
        self.info(f"📝 Logger initialized for experiment: {experiment_name}")
        self.info(f"📁 Log file: {self.log_file if file else 'Disabled'}")
        self.info(f"📊 TensorBoard: {tb_path if tensorboard else 'Disabled'}")
    
    # ==================== Basic Logging ====================
    
    def debug(self, message: str):
        """Log debug message"""
        logger.debug(message)
    
    def info(self, message: str):
        """Log info message"""
        logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        logger.critical(message)
    
    def success(self, message: str):
        """Log success message"""
        logger.success(message)
    
    # ==================== Training Logging ====================
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start"""
        self.epoch = epoch
        self.info(f"\n{'='*60}")
        self.info(f"🚀 Epoch {epoch}/{total_epochs} started")
        self.info(f"{'='*60}")
    
    def log_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch end with metrics"""
        self.info(f"\n📊 Epoch {epoch} completed:")
        for key, value in metrics.items():
            self.info(f"  {key}: {value:.4f}")
            if self.tensorboard_enabled:
                self.writer.add_scalar(f"epoch/{key}", value, epoch)
    
    def log_batch(self, step: int, loss: float, metrics: Optional[Dict[str, float]] = None):
        """Log batch training progress"""
        self.global_step = step
        
        msg = f"Step {step}: loss={loss:.4f}"
        if metrics:
            msg += ", " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.debug(msg)
        
        if self.tensorboard_enabled:
            self.writer.add_scalar("batch/loss", loss, step)
            if metrics:
                for key, value in metrics.items():
                    self.writer.add_scalar(f"batch/{key}", value, step)
    
    def log_validation(self, epoch: int, metrics: Dict[str, float]):
        """Log validation metrics"""
        self.info(f"\n✅ Validation results (Epoch {epoch}):")
        for key, value in metrics.items():
            self.info(f"  val_{key}: {value:.4f}")
            if self.tensorboard_enabled:
                self.writer.add_scalar(f"validation/{key}", value, epoch)
    
    # ==================== TDA Logging ====================
    
    def log_tda_computation(self, layer: int, computation_time: float, num_features: int):
        """Log TDA computation details"""
        self.debug(f"TDA Layer {layer}: {num_features} features extracted in {computation_time:.2f}s")
        if self.tensorboard_enabled:
            self.writer.add_scalar(f"tda/computation_time_layer_{layer}", computation_time, self.global_step)
    
    def log_tda_statistics(self, statistics: Dict[str, Any]):
        """Log TDA statistics"""
        self.info("📊 TDA Statistics:")
        for key, value in statistics.items():
            self.info(f"  {key}: {value}")
    
    # ==================== Data Augmentation Logging ====================
    
    def log_augmentation(self, technique: str, num_samples: int, avg_quality: float):
        """Log data augmentation progress"""
        self.info(f"🔄 {technique}: Generated {num_samples} samples (avg quality: {avg_quality:.3f})")
        if self.tensorboard_enabled:
            self.writer.add_scalar(f"augmentation/{technique}_quality", avg_quality, 0)
    
    # ==================== Checkpoint Logging ====================
    
    def log_checkpoint_save(self, epoch: int, path: str, is_best: bool = False):
        """Log checkpoint save"""
        prefix = "🌟 BEST" if is_best else "💾"
        self.info(f"{prefix} Checkpoint saved: {path} (Epoch {epoch})")
    
    def log_checkpoint_load(self, path: str, epoch: int):
        """Log checkpoint load"""
        self.success(f"✅ Resumed from checkpoint: {path} (Epoch {epoch})")
    
    # ==================== Error Tracking ====================
    
    def log_exception(self, exception: Exception, context: str = ""):
        """Log exception with context"""
        error_msg = f"❌ Exception in {context}: {type(exception).__name__}: {str(exception)}"
        self.error(error_msg)
        logger.exception(exception)  # This logs the full traceback
    
    # ==================== System Info ====================
    
    def log_system_info(self, info: Dict[str, Any]):
        """Log system information"""
        self.info("\n🖥️  System Information:")
        for key, value in info.items():
            self.info(f"  {key}: {value}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters"""
        self.info("\n⚙️  Hyperparameters:")
        for key, value in hparams.items():
            self.info(f"  {key}: {value}")
        
        if self.tensorboard_enabled:
            # TensorBoard hparams logging
            self.writer.add_hparams(hparams, {})
    
    # ==================== Experiment Summary ====================
    
    def log_experiment_summary(self, results: Dict[str, float], best_epoch: int):
        """Log final experiment summary"""
        self.info("\n" + "="*60)
        self.info("🎯 EXPERIMENT SUMMARY")
        self.info("="*60)
        self.info(f"Best Epoch: {best_epoch}")
        self.info("\nFinal Results:")
        for key, value in results.items():
            self.info(f"  {key}: {value:.4f}")
        self.info("="*60)
    
    # ==================== Progress Bar Integration ====================
    
    def get_tqdm_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for tqdm progress bar"""
        return {
            'file': sys.stdout,
            'dynamic_ncols': True,
            'colour': 'green'
        }
    
    # ==================== Cleanup ====================
    
    def close(self):
        """Close logger and TensorBoard writer"""
        if self.tensorboard_enabled and self.writer:
            self.writer.close()
        self.info("👋 Logger closed")


# ==================== Convenience Functions ====================

def get_logger(experiment_name: str, **kwargs) -> ExperimentLogger:
    """
    Factory function to create logger
    
    Args:
        experiment_name: Name of the experiment
        **kwargs: Additional arguments for ExperimentLogger
    
    Returns:
        ExperimentLogger instance
    """
    return ExperimentLogger(experiment_name, **kwargs)