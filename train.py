# train.py
import argparse
import os
import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from src.models.bert_classifier import SentimentClassifier
from src.models.trainer import SentimentTrainer
from src.data.preprocessor import create_data_loaders, load_data
from src.utils.metrics import ModelEvaluator, evaluate_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/reviews.csv',
                        help='Path to training data CSV')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test split ratio')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Validation split ratio')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='Pretrained BERT model name')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use mixed precision training')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Output directory for model checkpoints')
    parser.add_argument('--save_name', type=str, default='best_model.pt',
                        help='Name of saved model file')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("SENTIMENT ANALYSIS MODEL TRAINING")
    print("=" * 70)
    print(f"Data path: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Max length: {args.max_length}")
    print(f"Mixed precision: {args.use_amp}")
    print("=" * 70)
    
    # Load data
    print("\n[1/6] Loading data...")
    texts, labels = load_data(args.data_path)
    print(f"Loaded {len(texts)} samples")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Split data
    print("\n[2/6] Splitting data...")
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=args.test_size + args.val_size, 
        random_state=args.seed, stratify=labels
    )
    
    val_size_adjusted = args.val_size / (args.test_size + args.val_size)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=1-val_size_adjusted,
        random_state=args.seed, stratify=temp_labels
    )
    
    print(f"Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")
    
    # Initialize tokenizer
    print("\n[3/6] Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    
    # Create data loaders
    print("\n[4/6] Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        train_texts, train_labels,
        val_texts, val_labels,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Initialize model
    print("\n[5/6] Initializing model...")
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model = SentimentClassifier.load_from_checkpoint(args.resume)
    else:
        model = SentimentClassifier(
            n_classes=3,
            dropout=args.dropout,
            pretrained_model=args.model_name
        )
    
    # Initialize trainer
    trainer = SentimentTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        use_amp=args.use_amp
    )
    
    # Train model
    print("\n[6/6] Training model...")
    save_path = os.path.join(args.output_dir, args.save_name)
    history = trainer.train(save_path=save_path)
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("EVALUATING ON TEST SET")
    print("=" * 70)
    
    from torch.utils.data import DataLoader
    from src.data.preprocessor import SentimentDataset
    
    test_dataset = SentimentDataset(
        test_texts, test_labels, tokenizer, args.max_length
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    y_true, y_pred, y_proba = evaluate_model(model, test_loader, device)
    
    # Print evaluation results
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred)
    
    print("\nTest Set Performance:")
    print("-" * 70)
    for metric, value in metrics.items():
        print(f"{metric:25s}: {value:.4f}")
    
    evaluator.print_report(y_true, y_pred)
    
    # Save results
    results_path = os.path.join(args.output_dir, 'training_results.txt')
    with open(results_path, 'w') as f:
        f.write("Training Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Best Validation Accuracy: {trainer.best_val_acc:.4f}\n\n")
        f.write("Test Set Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print(f"\nResults saved to: {results_path}")
    print(f"Model saved to: {save_path}")
    print("\nTraining completed successfully! 🎉")

if __name__ == "__main__":
    main()
