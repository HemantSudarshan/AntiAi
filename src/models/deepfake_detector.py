"""
Deepfake Detection Model
Migrated from Jupyter notebook, with option to upgrade to EfficientNet-B4
"""
import numpy as np
import io
from typing import Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# Optional torch import
try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not installed - deepfake detection will run in demo mode")


class DeepfakeDetector:
    """
    Deepfake Detection using face detection + classification.
    
    Uses MTCNN for face detection and InceptionResnetV1 for classification.
    Can be upgraded to EfficientNet-B4 + Vision Transformer ensemble.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the deepfake detector.
        
        Args:
            model_path: Optional path to pre-trained weights
        """
        self._demo_mode = False
        self.mtcnn = None
        self.model = None
        self._last_prediction = None
        self._last_confidence = None
        self._last_face = None
        self.device = None
        
        if not TORCH_AVAILABLE:
            print("⚠️ PyTorch not available - running in demo mode")
            self._demo_mode = True
            return
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models(model_path)
    
    def _load_models(self, model_path: Optional[str] = None):
        """Load MTCNN and classifier models"""
        try:
            from facenet_pytorch import MTCNN, InceptionResnetV1
            
            # Face detection model
            self.mtcnn = MTCNN(
                select_largest=False,
                post_process=False,
                device=self.device
            ).eval()
            
            # Classification model
            self.model = InceptionResnetV1(
                pretrained="vggface2",
                classify=True,
                num_classes=1,
                device=self.device
            )
            
            # Load custom weights if available
            if model_path:
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✅ Loaded custom weights from {model_path}")
                except Exception as e:
                    print(f"⚠️ Could not load custom weights: {e}")
            
            self.model.to(self.device).eval()
            print("✅ Deepfake detection model loaded")
            
        except ImportError as e:
            print(f"⚠️ Could not import facenet_pytorch: {e}")
            print("   Running in demo mode")
            self._demo_mode = True
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            self._demo_mode = True
    
    def _preprocess_image(self, image_bytes: bytes):
        """
        Preprocess image for the model.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Preprocessed face tensor or None if no face detected
        """
        try:
            # Open image
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Detect face
            if self.mtcnn:
                face = self.mtcnn(img)
                if face is None:
                    print("⚠️ No face detected, using full image")
                    # Resize full image as fallback
                    img = img.resize((160, 160))
                    face = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
                    face = (face - 127.5) / 128.0
                
                self._last_face = face
                return face.unsqueeze(0).to(self.device)
            else:
                # Demo mode - return dummy tensor
                return torch.randn(1, 3, 160, 160).to(self.device)
                
        except Exception as e:
            print(f"⚠️ Image preprocessing error: {e}")
            return None
    
    def predict(self, image_bytes: bytes) -> int:
        """
        Predict if image is authentic (0) or deepfake (1).
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            0 for AUTHENTIC, 1 for DEEPFAKE
        """
        if hasattr(self, '_demo_mode') and self._demo_mode:
            # Demo mode - random prediction
            self._last_prediction = np.random.randint(0, 2)
            self._last_confidence = np.random.uniform(0.6, 0.95)
            return self._last_prediction
        
        face_tensor = self._preprocess_image(image_bytes)
        
        if face_tensor is None:
            raise ValueError("Could not process image")
        
        with torch.no_grad():
            output = self.model(face_tensor)
            prob = torch.sigmoid(output).item()
        
        self._last_confidence = prob if prob > 0.5 else 1 - prob
        self._last_prediction = 1 if prob > 0.5 else 0
        
        return self._last_prediction
    
    def get_confidence(self, image_bytes: bytes) -> float:
        """
        Get confidence score for the prediction.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Confidence score between 0 and 1
        """
        if self._last_confidence is None:
            self.predict(image_bytes)
        return self._last_confidence
    
    def get_face_tensor(self):
        """Get the last detected face tensor for visualization"""
        return self._last_face
    
    def get_model(self):
        """Get the underlying model for GradCAM"""
        return self.model
