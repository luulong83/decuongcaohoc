"""
Main Experiment Runner
Executes experiments E0-E4 with full pipeline
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from src.training.trainer import Trainer
from src.training.checkpoint import CheckpointManager
from src.data.dataset import load_uitvsfc_data
from src.models.fusion_model import create_model
from src.evaluation.metrics import compute_metrics


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run Vietnamese Text Classification Experiment")
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file (e.g., configs/experiment_configs/e4_proposed.yaml)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (auto will use CUDA if available)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        default='sentiment',
        choices=['sentiment', 'topic'],
        help='Task to run (sentiment or topic classification)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode (small subset of data)'
    )
    
    return parser.parse_args()


def setup_device(device_arg: str):
    """Setup computation device"""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'
    
    return torch.device(device)


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    """Main experiment execution"""
    
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = setup_device(args.device)
    print(f"\n🖥️  Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Set random seed
    seed = config.get('reproducibility', {}).get('seed', 42)
    set_seed(seed)
    print(f"🌱 Random seed: {seed}")
    
    # Create experiment name
    experiment_name = Path(args.config).stem
    if args.task != 'sentiment':
        experiment_name = f"{experiment_name}_{args.task}"
    
    # Initialize logger
    logger = get_logger(
        experiment_name=experiment_name,
        log_dir=config['paths']['log_dir'],
        level=config['logging']['level']
    )
    
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT: {experiment_name}")
    logger.info("=" * 60)
    
    # Log system info
    logger.log_system_info({
        'Python': sys.version.split()[0],
        'PyTorch': torch.__version__,
        'Device': str(device),
        'CUDA Available': torch.cuda.is_available(),
        'Working Directory': os.getcwd()
    })
    
    # Log hyperparameters
    logger.log_hyperparameters({
        'experiment': experiment_name,
        'task': args.task,
        'model': config['model']['name'],
        'use_tda': config['model']['tda']['enabled'],
        'use_augmentation': config['augmentation']['enabled'],
        'batch_size': config['dataset']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'num_epochs': config['training']['num_epochs']
    })
    
    try:
        # ==================== DATA LOADING ====================
        logger.info("\n📊 Loading dataset...")
        
        train_dataset, val_dataset, test_dataset = load_uitvsfc_data(
            data_dir=config['paths']['raw_data_dir'],
            task=args.task,
            max_length=config['dataset']['max_length'],
            use_augmentation=config['augmentation']['enabled'],
            augmentation_config=config['augmentation'] if config['augmentation']['enabled'] else None,
            debug=args.debug
        )
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        
        # ==================== MODEL CREATION ====================
        logger.info("\n🧠 Creating model...")
        
        # Determine number of classes
        num_classes = 3 if args.task == 'sentiment' else 10
        
        model_config = {
            'model_name': config['model']['name'],
            'num_classes': num_classes,
            'use_tda': config['model']['tda']['enabled'],
            'tda_feature_dim': config['model']['tda']['feature_dim'],
            'fusion_method': config['model']['fusion']['method'],
            'hidden_dims': config['model']['fusion']['hidden_dims'],
            'dropout': config['model']['fusion']['dropout']
        }
        
        model = create_model(model_config)
        model = model.to(device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # ==================== CHECKPOINT MANAGER ====================
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=config['paths']['checkpoint_dir'],
            experiment_name=experiment_name,
            keep_last_n=config['training']['checkpoint']['keep_last_n'],
            save_best=config['training']['checkpoint']['save_best']
        )
        
        # ==================== TRAINING ====================
        logger.info("\n🚀 Starting training...")
        
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            device=device,
            logger=logger,
            checkpoint_manager=checkpoint_manager
        )
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            logger.info(f"📥 Resuming from checkpoint: {args.resume}")
            checkpoint_info = checkpoint_manager.load_checkpoint(
                args.resume,
                model,
                trainer.optimizer,
                trainer.scheduler,
                device
            )
            start_epoch = checkpoint_info['epoch'] + 1
            logger.log_checkpoint_load(args.resume, checkpoint_info['epoch'])
        
        # Train
        best_val_metrics = trainer.train(start_epoch=start_epoch)
        
        # ==================== EVALUATION ====================
        logger.info("\n✅ Training completed. Evaluating on test set...")
        
        # Load best checkpoint
        best_checkpoint_path = checkpoint_manager.best_checkpoint_path
        if best_checkpoint_path:
            logger.info(f"Loading best checkpoint: {best_checkpoint_path}")
            checkpoint_manager.load_checkpoint(
                best_checkpoint_path,
                model,
                device=device
            )
        
        # Evaluate on test set
        test_metrics = trainer.evaluate(test_dataset, split='test')
        
        logger.info("\n📈 Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # ==================== SAVE RESULTS ====================
        results_dir = Path(config['paths']['results_dir']) / experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results to JSON
        import json
        results = {
            'experiment': experiment_name,
            'task': args.task,
            'config': model_config,
            'best_val_metrics': best_val_metrics,
            'test_metrics': test_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = results_dir / 'results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.success(f"✅ Results saved to: {results_file}")
        
        # ==================== SUMMARY ====================
        logger.log_experiment_summary(
            results=test_metrics,
            best_epoch=trainer.best_epoch
        )
        
        logger.success("🎉 Experiment completed successfully!")
        
    except Exception as e:
        logger.log_exception(e, context="main experiment")
        logger.error("❌ Experiment failed!")
        raise
    
    finally:
        # Cleanup
        logger.close()


if __name__ == "__main__":
    main()