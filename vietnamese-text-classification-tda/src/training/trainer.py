"""
Trainer - Main training loop with TDA integration
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, Any
import time

from src.tda.persistent_homology import PersistentHomologyComputer, extract_attention_maps
from src.tda.persistence_images import TDAFeatureExtractor
from src.evaluation.metrics import compute_metrics


class Trainer:
    """
    Training pipeline with:
    - TDA feature extraction
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Early stopping
    - Checkpoint management
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        config: Dict[str, Any],
        device: torch.device,
        logger,
        checkpoint_manager
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        
        # Training config
        self.num_epochs = config['training']['num_epochs']
        self.batch_size = config['dataset']['batch_size']
        self.gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
        self.max_grad_norm = config['training']['max_grad_norm']
        
        # TDA config
        self.use_tda = config['model']['tda']['enabled']
        if self.use_tda:
            self.ph_computer = PersistentHomologyComputer(
                selected_layers=config['model']['tda']['selected_layers'],
                homology_dims=config['model']['tda']['homology_dims'],
                max_dimension=config['model']['tda']['max_dimension']
            )
            self.tda_vectorizer = TDAFeatureExtractor(
                selected_layers=config['model']['tda']['selected_layers'],
                homology_dims=config['model']['tda']['homology_dims'],
                resolution=(
                    config['model']['tda']['persistence_image']['resolution'],
                    config['model']['tda']['persistence_image']['resolution']
                ),
                sigma=config['model']['tda']['persistence_image']['sigma'],
                normalize=config['model']['tda']['persistence_image']['normalize']
            )
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config['dataset']['num_workers'],
            pin_memory=True if device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=config['dataset']['num_workers'],
            pin_memory=True if device.type == 'cuda' else False
        )
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=config['training']['optimizer']['betas'],
            eps=config['training']['optimizer']['eps']
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * self.num_epochs // self.gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config['training']['scheduler']['num_warmup_steps'],
            num_training_steps=total_steps
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision
        self.use_amp = config['computation']['mixed_precision']
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Early stopping
        self.early_stopping_enabled = config['training']['early_stopping']['enabled']
        self.patience = config['training']['early_stopping']['patience']
        self.min_delta = config['training']['early_stopping']['min_delta']
        self.monitor_metric = config['training']['early_stopping']['monitor']
        self.best_score = float('-inf')
        self.patience_counter = 0
        self.best_epoch = 0
        
        # Tracking
        self.global_step = 0
        self.train_losses = []
        self.val_metrics_history = []
    
    def train(self, start_epoch: int = 0) -> Dict[str, float]:
        """
        Main training loop
        
        Args:
            start_epoch: Epoch to start from (for resuming)
        
        Returns:
            Best validation metrics
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info("TRAINING STARTED")
        self.logger.info(f"{'='*60}")
        
        for epoch in range(start_epoch, self.num_epochs):
            self.logger.log_epoch_start(epoch + 1, self.num_epochs)
            
            # Train one epoch
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.evaluate(self.val_dataset, split='val')
            
            # Log epoch results
            epoch_metrics = {
                'train_loss': train_loss,
                'train_acc': train_metrics.get('accuracy', 0),
                'val_loss': val_metrics.get('loss', 0),
                'val_acc': val_metrics.get('accuracy', 0),
                'val_f1': val_metrics.get('f1_macro', 0)
            }
            self.logger.log_epoch_end(epoch + 1, epoch_metrics)
            self.logger.log_validation(epoch + 1, val_metrics)
            
            # Track history
            self.val_metrics_history.append(val_metrics)
            
            # Check if best model
            current_score = val_metrics.get(self.monitor_metric, 0)
            is_best = current_score > self.best_score
            
            if is_best:
                self.best_score = current_score
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                self.logger.success(f"🌟 New best {self.monitor_metric}: {current_score:.4f}")
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch + 1,
                global_step=self.global_step,
                metrics=val_metrics,
                scheduler=self.scheduler,
                is_best=is_best
            )
            self.logger.log_checkpoint_save(epoch + 1, checkpoint_path, is_best)
            
            # Early stopping
            if self.early_stopping_enabled:
                if self.patience_counter >= self.patience:
                    self.logger.warning(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Load best model
        best_metrics = self.val_metrics_history[self.best_epoch - 1]
        return best_metrics
    
    def train_epoch(self, epoch: int) -> tuple:
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Training Epoch {epoch + 1}",
            **self.logger.get_tqdm_kwargs()
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                if self.use_tda:
                    # Extract TDA features
                    tda_features = self._extract_tda_features_batch(input_ids, attention_mask)
                    outputs = self.model(input_ids, attention_mask, tda_features)
                    logits = outputs['logits']
                else:
                    logits = self.model(input_ids, attention_mask)
                
                loss = self.criterion(logits, labels)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Track metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log batch
            if self.global_step % self.config['logging']['log_interval'] == 0:
                self.logger.log_batch(
                    self.global_step,
                    loss.item() * self.gradient_accumulation_steps
                )
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        train_metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
        
        return avg_loss, train_metrics
    
    def evaluate(self, dataset, split: str = 'val') -> Dict[str, float]:
        """Evaluate model on dataset"""
        self.model.eval()
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config['dataset']['num_workers']
        )
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Evaluating ({split})",
            **self.logger.get_tqdm_kwargs()
        )
        
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.use_tda:
                    tda_features = self._extract_tda_features_batch(input_ids, attention_mask)
                    outputs = self.model(input_ids, attention_mask, tda_features)
                    logits = outputs['logits']
                else:
                    logits = self.model(input_ids, attention_mask)
                
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(dataloader)
        metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
        metrics['loss'] = avg_loss
        
        return metrics
    
    def _extract_tda_features_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract TDA features for a batch"""
        batch_size = input_ids.shape[0]
        tda_features_list = []
        
        # Get attention maps
        with torch.no_grad():
            outputs = self.model.phobert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
        
        # Process each sample in batch
        for i in range(batch_size):
            # Extract attention for this sample
            sample_attentions = extract_attention_maps(
                outputs,
                selected_layers=self.config['model']['tda']['selected_layers']
            )[i:i+1]  # Keep batch dimension
            
            # Compute persistent homology
            start_time = time.time()
            persistence_diagrams = self.ph_computer.compute_from_attention(
                sample_attentions.squeeze(0)
            )
            computation_time = time.time() - start_time
            
            # Vectorize to features
            tda_features = self.tda_vectorizer.transform(persistence_diagrams)
            tda_features_list.append(tda_features)
            
            # Log TDA computation
            if i == 0:  # Log only first sample to avoid spam
                self.logger.log_tda_computation(
                    layer=self.config['model']['tda']['selected_layers'][0],
                    computation_time=computation_time,
                    num_features=len(tda_features)
                )
        
        # Stack to tensor
        tda_features_tensor = torch.tensor(
            np.array(tda_features_list),
            dtype=torch.float32,
            device=self.device
        )
        
        return tda_features_tensor