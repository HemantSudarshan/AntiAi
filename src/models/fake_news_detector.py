"""
Fake News Detection Model - Phase 3 Enhanced Version
Uses TF-IDF + Ensemble of classifiers with optional transformer support
Includes LIME explainability integration
"""
import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Callable
import warnings
warnings.filterwarnings("ignore")

# Check for optional dependencies
SKLEARN_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
LIME_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-learn not installed")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ transformers not installed")

try:
    import lime.lime_text
    LIME_AVAILABLE = True
except ImportError:
    print("⚠️ LIME not installed")


class FakeNewsDetector:
    """
    Fake News Detection using ensemble of classifiers.
    
    Phase 3 Features:
    - Ensemble voting (Logistic Regression + Random Forest + Gradient Boosting)
    - LIME explainability integration
    - Optional transformer support (RoBERTa)
    """
    
    def __init__(self, use_transformers: bool = False, use_ensemble: bool = True):
        """
        Initialize the fake news detector.
        
        Args:
            use_transformers: If True, use RoBERTa. If False, use ensemble.
            use_ensemble: If True, use multiple classifiers with voting.
        """
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE
        self.use_ensemble = use_ensemble and SKLEARN_AVAILABLE
        self.models = {}
        self.vectorizer = None
        self._last_prediction = None
        self._last_confidence = None
        self._last_probabilities = None
        self._is_fitted = False
        
        # LIME explainer
        self.lime_explainer = None
        if LIME_AVAILABLE:
            self.lime_explainer = lime.lime_text.LimeTextExplainer(
                class_names=['REAL', 'FAKE']
            )
        
        self._load_models()
    
    def _load_models(self):
        """Load the classification models"""
        if self.use_transformers:
            self._load_transformer_model()
        elif self.use_ensemble:
            self._load_ensemble_models()
        else:
            self._load_simple_model()
    
    def _load_simple_model(self):
        """Load a simple TF-IDF + Logistic Regression model"""
        if not SKLEARN_AVAILABLE:
            print("⚠️ scikit-learn not available, using heuristics")
            return
            
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        self.models['logistic'] = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        
        print("✅ Simple fake news model loaded")
    
    def _load_ensemble_models(self):
        """Load ensemble of classifiers for better accuracy"""
        if not SKLEARN_AVAILABLE:
            print("⚠️ scikit-learn not available")
            return
            
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        # Ensemble of classifiers
        self.models = {
            'logistic': LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                C=1.0
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        }
        
        print("✅ Ensemble models loaded (LR + RF + GB)")
    
    def _load_transformer_model(self):
        """Load RoBERTa-based model for better accuracy"""
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️ Transformers not available, falling back to ensemble")
            self.use_transformers = False
            self._load_ensemble_models()
            return
            
        try:
            model_name = "cross-encoder/qnli-distilroberta-base"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.transformer.eval()
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.transformer.to(self.device)
            
            print(f"✅ Transformer model loaded on {self.device}")
        except Exception as e:
            print(f"⚠️ Could not load transformer model: {e}")
            self.use_transformers = False
            self._load_ensemble_models()
    
    def fit(self, texts: List[str], labels: List[int]):
        """
        Train the model on labeled data.
        
        Args:
            texts: List of news article texts
            labels: List of labels (0=REAL, 1=FAKE)
        """
        if not SKLEARN_AVAILABLE or self.use_transformers:
            return
            
        # Fit vectorizer
        X = self.vectorizer.fit_transform(texts)
        
        # Fit each model in ensemble
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X, labels)
        
        self._is_fitted = True
        print("✅ Models trained successfully")
    
    def predict(self, text: str) -> int:
        """
        Predict if text is fake (1) or real (0).
        
        Args:
            text: News article text
            
        Returns:
            0 for REAL, 1 for FAKE
        """
        if self.use_transformers:
            return self._predict_transformer(text)
        elif self._is_fitted and SKLEARN_AVAILABLE:
            return self._predict_ensemble(text)
        else:
            return self._predict_heuristic(text)
    
    def _predict_transformer(self, text: str) -> int:
        """Predict using transformer model"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.transformer(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            self._last_probabilities = probs[0].cpu().numpy()
        
        self._last_prediction = int(probs[0][1] > 0.5)
        self._last_confidence = float(max(probs[0]))
        return self._last_prediction
    
    def _predict_ensemble(self, text: str) -> int:
        """Predict using ensemble of classifiers with voting"""
        X = self.vectorizer.transform([text])
        
        predictions = []
        probabilities = []
        
        for name, model in self.models.items():
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0]
            predictions.append(pred)
            probabilities.append(prob)
        
        # Soft voting (average probabilities)
        avg_probs = np.mean(probabilities, axis=0)
        self._last_probabilities = avg_probs
        self._last_prediction = int(avg_probs[1] > 0.5)
        self._last_confidence = float(max(avg_probs))
        
        return self._last_prediction
    
    def _predict_heuristic(self, text: str) -> int:
        """Fallback heuristic-based prediction"""
        text_lower = text.lower()
        
        fake_indicators = [
            "breaking", "urgent", "shocking", "unbelievable",
            "you won't believe", "secret", "they don't want you to know",
            "mainstream media", "cover up", "conspiracy",
            "miracle", "cure", "100%", "guaranteed"
        ]
        
        real_indicators = [
            "according to", "research shows", "study finds",
            "officials said", "reported", "confirmed",
            "source:", "reuters", "associated press"
        ]
        
        fake_count = sum(1 for ind in fake_indicators if ind in text_lower)
        real_count = sum(1 for ind in real_indicators if ind in text_lower)
        
        score = fake_count - real_count
        
        # Additional heuristics
        if text.count('!') > 3 or text.count('?') > 3:
            score += 1
        
        caps_words = sum(1 for w in text.split() if w.isupper() and len(w) > 2)
        if caps_words > 5:
            score += 1
        
        if len(text) < 100:
            score += 0.5
        
        confidence = min(0.5 + abs(score) * 0.1, 0.99)
        
        self._last_prediction = 1 if score > 0 else 0
        self._last_confidence = confidence
        self._last_probabilities = np.array([1 - confidence, confidence]) if self._last_prediction == 1 else np.array([confidence, 1 - confidence])
        
        return self._last_prediction
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get probability predictions for multiple texts.
        Required for LIME integration.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of shape (n_samples, 2) with probabilities
        """
        probas = []
        for text in texts:
            self.predict(text)
            probas.append(self._last_probabilities)
        return np.array(probas)
    
    def get_confidence(self, text: str) -> float:
        """Get confidence score for the prediction."""
        if self._last_confidence is None:
            self.predict(text)
        return self._last_confidence
    
    def get_lime_explanation(self, text: str, num_features: int = 10) -> Dict[str, Any]:
        """
        Get LIME explanation for the prediction.
        
        Args:
            text: Input text to explain
            num_features: Number of top features to return
            
        Returns:
            Dictionary with explanation data
        """
        if not LIME_AVAILABLE or self.lime_explainer is None:
            return self._get_heuristic_explanation(text)
        
        try:
            # Get the explanation
            exp = self.lime_explainer.explain_instance(
                text,
                self.predict_proba,
                num_features=num_features,
                num_samples=500
            )
            
            # Extract feature importances
            features = []
            for word, importance in exp.as_list():
                features.append({
                    'word': word,
                    'importance': float(importance),
                    'impact': 'increases fake probability' if importance > 0 else 'increases real probability'
                })
            
            return {
                'method': 'LIME',
                'top_features': features,
                'prediction': self._last_prediction,
                'confidence': self._last_confidence
            }
            
        except Exception as e:
            print(f"⚠️ LIME error: {e}")
            return self._get_heuristic_explanation(text)
    
    def _get_heuristic_explanation(self, text: str) -> Dict[str, Any]:
        """Fallback heuristic-based explanation"""
        text_lower = text.lower()
        
        fake_keywords = ["breaking", "urgent", "shocking", "secret", "conspiracy"]
        real_keywords = ["according to", "research", "study", "officials", "confirmed"]
        
        features = []
        for kw in fake_keywords:
            if kw in text_lower:
                features.append({'word': kw, 'importance': 0.8, 'impact': 'fake indicator'})
        for kw in real_keywords:
            if kw in text_lower:
                features.append({'word': kw, 'importance': -0.8, 'impact': 'real indicator'})
        
        return {
            'method': 'Heuristic',
            'top_features': features[:10],
            'prediction': self._last_prediction,
            'confidence': self._last_confidence
        }
