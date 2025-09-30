# predict.py
"""
Inference script for sentiment analysis.
Usage: 
    python predict.py --text "This product is amazing!"
    python predict.py --file data/test_reviews.csv
"""

import argparse
import torch
import pandas as pd
from transformers import BertTokenizer
from typing import List, Tuple

from src.models.bert_classifier import SentimentClassifier

class SentimentPredictor:
    """Easy-to-use predictor class."""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to run inference on
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['Negative', 'Neutral', 'Positive']
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load model
        self.model = SentimentClassifier.load_from_checkpoint(
            model_path, 
            device=self.device
        )
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
    
    def predict_single(self, text: str) -> Tuple[str, float, dict]:
        """
        Predict sentiment for a single text.
        
        Returns:
            Tuple of (sentiment, confidence, probabilities)
        """
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions, probabilities = self.model.predict(input_ids, attention_mask)
        
        # Get results
        pred_idx = predictions[0].item()
        confidence = probabilities[0][pred_idx].item()
        
        prob_dict = {
            self.class_names[i]: float(probabilities[0][i])
            for i in range(len(self.class_names))
        }
        
        return self.class_names[pred_idx], confidence, prob_dict
    
    def predict_batch(self, texts: List[str]) -> List[dict]:
        """
        Predict sentiment for multiple texts.
        
        Returns:
            List of prediction dictionaries
        """
        # Tokenize batch
        encoding = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions, probabilities = self.model.predict(input_ids, attention_mask)
        
        # Process results
        results = []
        for i, text in enumerate(texts):
            pred_idx = predictions[i].item()
            confidence = probabilities[i][pred_idx].item()
            
            prob_dict = {
                self.class_names[j]: float(probabilities[i][j])
                for j in range(len(self.class_names))
            }
            
            results.append({
                'text': text,
                'sentiment': self.class_names[pred_idx],
                'confidence': confidence,
                'probabilities': prob_dict
            })
        
        return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict sentiment')
    
    parser.add_argument('--model_path', type=str, default='models/best_model.pt',
                        help='Path to trained model')
    parser.add_argument('--text', type=str, default=None,
                        help='Single text to predict')
    parser.add_argument('--file', type=str, default=None,
                        help='CSV file with texts to predict')
    parser.add_argument('--text_column', type=str, default='text',
                        help='Name of text column in CSV')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output file for batch predictions')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for file processing')
    
    return parser.parse_args()

def main():
    """Main prediction function."""
    args = parse_args()
    
    # Initialize predictor
    print("Loading model...")
    predictor = SentimentPredictor(args.model_path)
    
    if args.text:
        # Single text prediction
        print("\n" + "=" * 70)
        print("SINGLE TEXT PREDICTION")
        print("=" * 70)
        print(f"Text: {args.text}")
        print("-" * 70)
        
        sentiment, confidence, probabilities = predictor.predict_single(args.text)
        
        print(f"\nPrediction: {sentiment}")
        print(f"Confidence: {confidence:.4f}")
        print("\nProbabilities:")
        for label, prob in probabilities.items():
            bar = "█" * int(prob * 50)
            print(f"  {label:10s}: {prob:.4f} {bar}")
    
    elif args.file:
        # Batch file prediction
        print("\n" + "=" * 70)
        print("BATCH FILE PREDICTION")
        print("=" * 70)
        print(f"Input file: {args.file}")
        
        # Load data
        df = pd.read_csv(args.file)
        texts = df[args.text_column].astype(str).tolist()
        
        print(f"Processing {len(texts)} texts...")
        
        # Process in batches
        all_results = []
        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i:i + args.batch_size]
            results = predictor.predict_batch(batch_texts)
            all_results.extend(results)
            print(f"Processed {min(i + args.batch_size, len(texts))}/{len(texts)}")
        
        # Create results dataframe
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(args.output, index=False)
        
        print(f"\nPredictions saved to: {args.output}")
        
        # Print summary
        print("\nSentiment Distribution:")
        print(results_df['sentiment'].value_counts())
        print(f"\nAverage Confidence: {results_df['confidence'].mean():.4f}")
    
    else:
        print("Error: Please provide either --text or --file argument")
        print("Examples:")
        print("  python predict.py --text 'Great product!'")
        print("  python predict.py --file data/reviews.csv")

if __name__ == "__main__":
    main()
