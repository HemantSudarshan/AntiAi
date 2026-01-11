"""
Model package initialization
"""
from .fake_news_detector import FakeNewsDetector
from .deepfake_detector import DeepfakeDetector

__all__ = ["FakeNewsDetector", "DeepfakeDetector"]
