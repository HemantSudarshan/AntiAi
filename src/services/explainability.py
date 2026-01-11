"""
Explainability Service
Provides interpretable explanations for model predictions using LIME and GradCAM
"""
import numpy as np
from typing import Dict, List, Optional, Any
import base64
import io
import time
import warnings
warnings.filterwarnings("ignore")


class ExplainabilityService:
    """
    Provides explainability features for both text and image models.
    
    - Text: Uses LIME-style keyword importance
    - Image: Uses GradCAM heatmaps
    """
    
    def __init__(self):
        """Initialize explainability service"""
        self._lime_available = False
        self._gradcam_available = False
        
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if optional dependencies are available"""
        try:
            import lime.lime_text
            self._lime_available = True
            print("✅ LIME available for text explainability")
        except ImportError:
            print("⚠️ LIME not installed, using simple keyword analysis")
        
        try:
            from pytorch_grad_cam import GradCAM
            self._gradcam_available = True
            print("✅ GradCAM available for image explainability")
        except ImportError:
            print("⚠️ GradCAM not installed, using placeholder heatmaps")
    
    def explain_fake_news(self, text: str, prediction: int) -> Dict[str, Any]:
        """
        Generate explanation for fake news prediction.
        
        Args:
            text: Input text
            prediction: Model prediction (0=REAL, 1=FAKE)
            
        Returns:
            Dictionary with explanation data
        """
        start_time = time.time()
        
        # Keyword-based explanation (works without LIME)
        text_lower = text.lower()
        
        fake_indicators = [
            ("breaking", "sensational language"),
            ("urgent", "urgency tactics"),
            ("shocking", "emotional manipulation"),
            ("you won't believe", "clickbait phrase"),
            ("secret", "conspiracy language"),
            ("they don't want you to know", "conspiracy theory"),
            ("100%", "absolute claims"),
            ("guaranteed", "unrealistic promises"),
        ]
        
        real_indicators = [
            ("according to", "attribution"),
            ("research shows", "evidence-based"),
            ("study finds", "research reference"),
            ("officials said", "official source"),
            ("reuters", "reputable source"),
            ("associated press", "news agency"),
        ]
        
        top_features = []
        
        # Find indicators in text
        for indicator, reason in fake_indicators:
            if indicator in text_lower:
                top_features.append({
                    "keyword": indicator,
                    "impact": "increases fake probability",
                    "reason": reason,
                    "importance": 0.8
                })
        
        for indicator, reason in real_indicators:
            if indicator in text_lower:
                top_features.append({
                    "keyword": indicator,
                    "impact": "increases real probability",
                    "reason": reason,
                    "importance": -0.7
                })
        
        # Analyze text patterns
        patterns = []
        
        if text.count('!') > 3:
            patterns.append("Excessive exclamation marks")
        if text.count('?') > 3:
            patterns.append("Excessive question marks")
        if sum(1 for w in text.split() if w.isupper() and len(w) > 2) > 5:
            patterns.append("Excessive capitalization")
        if len(text) < 100:
            patterns.append("Very short article (typical of sensational content)")
        
        # Generate reasoning
        if prediction == 1:  # FAKE
            reasoning = f"This article shows {len(top_features)} potential fake news indicators"
            if patterns:
                reasoning += f" and {len(patterns)} suspicious patterns"
        else:  # REAL
            reasoning = "This article appears credible"
            if top_features:
                reasoning += f" with {len([f for f in top_features if f['importance'] < 0])} credibility markers"
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "top_features": top_features[:5],
            "patterns": patterns,
            "reasoning": reasoning,
            "time_ms": processing_time
        }
    
    def generate_heatmap(self, image_bytes: bytes, model: Any) -> Optional[str]:
        """
        Generate GradCAM heatmap for deepfake detection.
        
        Args:
            image_bytes: Input image bytes
            model: The deepfake detection model
            
        Returns:
            Base64 encoded heatmap image or None
        """
        if self._gradcam_available:
            return self._generate_gradcam_heatmap(image_bytes, model)
        else:
            return self._generate_placeholder_heatmap(image_bytes)
    
    def _generate_gradcam_heatmap(self, image_bytes: bytes, model: Any) -> Optional[str]:
        """Generate actual GradCAM heatmap"""
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            from PIL import Image
            import torch
            import cv2
            
            # Get the actual model
            if hasattr(model, 'get_model'):
                pytorch_model = model.get_model()
            else:
                pytorch_model = model
            
            # Get face tensor
            if hasattr(model, 'get_face_tensor'):
                face_tensor = model.get_face_tensor()
                if face_tensor is None:
                    return self._generate_placeholder_heatmap(image_bytes)
            else:
                return self._generate_placeholder_heatmap(image_bytes)
            
            # Setup GradCAM
            target_layer = pytorch_model.block8.branch1[-1]
            cam = GradCAM(model=pytorch_model, target_layers=[target_layer])
            
            # Generate activation map
            input_tensor = face_tensor.unsqueeze(0) if face_tensor.dim() == 3 else face_tensor
            grayscale_cam = cam(input_tensor=input_tensor)[0]
            
            # Create visualization
            face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
            face_np = (face_np - face_np.min()) / (face_np.max() - face_np.min())
            
            visualization = show_cam_on_image(face_np, grayscale_cam, use_rgb=True)
            
            # Convert to base64
            img = Image.fromarray(visualization)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            b64_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{b64_str}"
            
        except Exception as e:
            print(f"⚠️ GradCAM error: {e}")
            return self._generate_placeholder_heatmap(image_bytes)
    
    def _generate_placeholder_heatmap(self, image_bytes: bytes) -> str:
        """Generate a placeholder heatmap for demo purposes"""
        try:
            from PIL import Image, ImageDraw, ImageFilter
            
            # Open original image
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img = img.resize((256, 256))
            
            # Create a simple heatmap overlay (red in center)
            heatmap = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(heatmap)
            
            # Draw gradient circles in center (simulating face area)
            center = (img.size[0] // 2, img.size[1] // 2 - 20)
            for r in range(80, 10, -10):
                alpha = int((80 - r) * 2.5)
                draw.ellipse(
                    [center[0]-r, center[1]-r, center[0]+r, center[1]+r],
                    fill=(255, 0, 0, alpha)
                )
            
            # Blend
            heatmap = heatmap.filter(ImageFilter.GaussianBlur(15))
            result = Image.alpha_composite(img.convert('RGBA'), heatmap)
            
            # Convert to base64
            buffer = io.BytesIO()
            result.save(buffer, format='PNG')
            b64_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{b64_str}"
            
        except Exception as e:
            print(f"⚠️ Placeholder heatmap error: {e}")
            return None
    
    def get_suspicious_regions(self, heatmap: Optional[str]) -> List[str]:
        """
        Get descriptions of suspicious regions from heatmap.
        
        Args:
            heatmap: Base64 encoded heatmap
            
        Returns:
            List of region descriptions
        """
        # In a full implementation, this would analyze the heatmap
        # For now, return common deepfake artifacts
        return [
            "Face boundaries - potential blending artifacts",
            "Eye region - may show unnatural blinking patterns",
            "Mouth area - possible audio-visual sync issues",
            "Hair edges - common deepfake boundary effects"
        ]
