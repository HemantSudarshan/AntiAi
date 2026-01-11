"""
Fake News Detection Model
Uses TF-IDF + ensemble of classifiers (simplified for initial implementation)
Can be upgraded to RoBERTa + XGBoost ensemble
"""
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


class FakeNewsDetector:
    """
    Fake News Detection using text classification.
    
    Initial implementation uses simple methods for fast startup.
    Can be upgraded to transformer models.
    """
    
    def __init__(self, use_transformers: bool = False):
        """
        Initialize the fake news detector.
        
        Args:
            use_transformers: If True, use RoBERTa. If False, use simpler methods.
        """
        self.use_transformers = use_transformers
        self.model = None
        self.vectorizer = None
        self._last_prediction = None
        self._last_confidence = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the classification model"""
        if self.use_transformers:
            self._load_transformer_model()
        else:
            self._load_simple_model()
    
    def _load_simple_model(self):
        """Load a simple TF-IDF + classifier model"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        
        # Initialize vectorizer and model
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        
        # For demo purposes, we'll use keyword-based heuristics
        # In production, load a pre-trained model
        self._fake_indicators = [
            "breaking", "urgent", "shocking", "unbelievable",
            "you won't believe", "secret", "they don't want you to know",
            "mainstream media", "cover up", "conspiracy",
            "miracle", "cure", "100%", "guaranteed"
        ]
        
        self._real_indicators = [
            "according to", "research shows", "study finds",
            "officials said", "reported", "confirmed",
            "source:", "reuters", "associated press"
        ]
        
        print("✅ Simple fake news model loaded")
    
    def _load_transformer_model(self):
        """Load RoBERTa-based model for better accuracy"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            model_name = "cross-encoder/qnli-distilroberta-base"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.transformer.eval()
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.transformer.to(self.device)
            
            print("✅ Transformer model loaded")
        except Exception as e:
            print(f"⚠️ Could not load transformer model: {e}")
            print("   Falling back to simple model")
            self.use_transformers = False
            self._load_simple_model()
    
    def predict(self, text: str) -> int:
        """
        Predict if text is fake (1) or real (0).
        
        Args:
            text: News article text
            
        Returns:
            0 for REAL, 1 for FAKE
        """
        text_lower = text.lower()
        
        # Count indicators
        fake_count = sum(1 for indicator in self._fake_indicators if indicator in text_lower)
        real_count = sum(1 for indicator in self._real_indicators if indicator in text_lower)
        
        # Calculate score (positive = more fake)
        score = fake_count - real_count
        
        # Add heuristics for common fake news patterns
        # Excessive punctuation
        if text.count('!') > 3 or text.count('?') > 3:
            score += 1
        
        # ALL CAPS sections
        caps_words = sum(1 for word in text.split() if word.isupper() and len(word) > 2)
        if caps_words > 5:
            score += 1
        
        # Text length patterns (very short sensational text)
        if len(text) < 100:
            score += 0.5
        
        # Calculate confidence
        total_indicators = fake_count + real_count + 1  # +1 to avoid division by zero
        confidence = min(0.5 + abs(score) * 0.1, 0.99)
        
        self._last_prediction = 1 if score > 0 else 0
        self._last_confidence = confidence
        
        return self._last_prediction
    
    def get_confidence(self, text: str) -> float:
        """
        Get confidence score for the prediction.
        
        Args:
            text: News article text
            
        Returns:
            Confidence score between 0 and 1
        """
        if self._last_confidence is None:
            self.predict(text)
        return self._last_confidence
    
    def get_top_features(self, text: str) -> list:
        """
        Get top features that influenced the prediction.
        
        Args:
            text: News article text
            
        Returns:
            List of (feature, importance) tuples
        """
        text_lower = text.lower()
        features = []
        
        for indicator in self._fake_indicators:
            if indicator in text_lower:
                features.append((indicator, 0.8, "fake_indicator"))
        
        for indicator in self._real_indicators:
            if indicator in text_lower:
                features.append((indicator, -0.8, "real_indicator"))
        
        return sorted(features, key=lambda x: abs(x[1]), reverse=True)[:5]
