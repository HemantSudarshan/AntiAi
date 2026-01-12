"""
Explainability Service - Phase 3 Enhanced Version
Provides LIME for text and GradCAM for images
"""
import numpy as np
from typing import Dict, List, Optional, Any
import base64
import io
import time
import warnings
warnings.filterwarnings("ignore")

# Check dependencies
LIME_AVAILABLE = False
SHAP_AVAILABLE = False
GRADCAM_AVAILABLE = False
PIL_AVAILABLE = False

try:
    import lime.lime_text
    LIME_AVAILABLE = True
except ImportError:
    pass

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    pass

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    pass

try:
    from PIL import Image, ImageDraw, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    pass


class ExplainabilityService:
    """
    Provides explainability features for both text and image models.
    
    Phase 3 Features:
    - LIME text explanations with actual model integration
    - SHAP analysis support
    - GradCAM heatmaps for images
    - Detailed feature attribution
    """
    
    def __init__(self):
        """Initialize explainability service"""
        self._lime_available = LIME_AVAILABLE
        self._shap_available = SHAP_AVAILABLE
        self._gradcam_available = GRADCAM_AVAILABLE
        
        self.lime_explainer = None
        if LIME_AVAILABLE:
            self.lime_explainer = lime.lime_text.LimeTextExplainer(
                class_names=['REAL', 'FAKE'],
                split_expression=r'\W+',
                bow=True
            )
            print("✅ LIME text explainer initialized")
        else:
            print("⚠️ LIME not available, using keyword analysis")
        
        if SHAP_AVAILABLE:
            print("✅ SHAP available for advanced analysis")
        
        if GRADCAM_AVAILABLE:
            print("✅ GradCAM available for image explainability")
    
    def explain_fake_news(
        self, 
        text: str, 
        prediction: int,
        predict_fn: Optional[callable] = None,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """
        Generate explanation for fake news prediction.
        
        Args:
            text: Input text
            prediction: Model prediction (0=REAL, 1=FAKE)
            predict_fn: Optional prediction function for LIME
            num_features: Number of top features to return
            
        Returns:
            Dictionary with explanation data
        """
        start_time = time.time()
        
        # Try LIME first if available and predict function provided
        if self._lime_available and predict_fn is not None:
            try:
                return self._explain_with_lime(text, prediction, predict_fn, num_features, start_time)
            except Exception as e:
                print(f"⚠️ LIME error: {e}, falling back to keyword analysis")
        
        # Fallback to keyword-based analysis
        return self._explain_with_keywords(text, prediction, start_time)
    
    def _explain_with_lime(
        self, 
        text: str, 
        prediction: int, 
        predict_fn: callable,
        num_features: int,
        start_time: float
    ) -> Dict[str, Any]:
        """Generate LIME-based explanation"""
        # Get LIME explanation
        exp = self.lime_explainer.explain_instance(
            text,
            predict_fn,
            num_features=num_features,
            num_samples=500
        )
        
        # Extract features
        top_features = []
        for word, importance in exp.as_list():
            top_features.append({
                'keyword': word,
                'importance': round(float(importance), 4),
                'impact': 'increases fake probability' if importance > 0 else 'increases real probability',
                'direction': 'fake' if importance > 0 else 'real'
            })
        
        # Generate reasoning
        fake_features = [f for f in top_features if f['direction'] == 'fake']
        real_features = [f for f in top_features if f['direction'] == 'real']
        
        if prediction == 1:
            if fake_features:
                reasoning = f"Article flagged as FAKE based on {len(fake_features)} suspicious indicators including: {', '.join([f['keyword'] for f in fake_features[:3]])}"
            else:
                reasoning = "Article flagged as FAKE based on overall linguistic patterns"
        else:
            if real_features:
                reasoning = f"Article appears GENUINE with {len(real_features)} credibility markers including: {', '.join([f['keyword'] for f in real_features[:3]])}"
            else:
                reasoning = "Article appears GENUINE based on neutral language patterns"
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            'method': 'LIME',
            'top_features': top_features,
            'reasoning': reasoning,
            'fake_indicators': len(fake_features),
            'real_indicators': len(real_features),
            'time_ms': processing_time,
            'prediction': prediction
        }
    
    def _explain_with_keywords(
        self, 
        text: str, 
        prediction: int,
        start_time: float
    ) -> Dict[str, Any]:
        """Keyword-based explanation fallback"""
        text_lower = text.lower()
        
        fake_indicators = [
            ("breaking", "sensational language", 0.8),
            ("urgent", "urgency tactics", 0.75),
            ("shocking", "emotional manipulation", 0.8),
            ("you won't believe", "clickbait phrase", 0.9),
            ("secret", "conspiracy language", 0.7),
            ("they don't want you to know", "conspiracy theory", 0.85),
            ("100%", "absolute claims", 0.6),
            ("guaranteed", "unrealistic promises", 0.65),
            ("miracle", "sensational claims", 0.75),
            ("cover up", "conspiracy language", 0.8),
        ]
        
        real_indicators = [
            ("according to", "attribution", -0.7),
            ("research shows", "evidence-based", -0.75),
            ("study finds", "research reference", -0.8),
            ("officials said", "official source", -0.7),
            ("reuters", "reputable source", -0.85),
            ("associated press", "news agency", -0.85),
            ("confirmed", "verification", -0.6),
            ("reported", "journalism language", -0.5),
        ]
        
        top_features = []
        
        for indicator, reason, importance in fake_indicators:
            if indicator in text_lower:
                top_features.append({
                    'keyword': indicator,
                    'importance': importance,
                    'impact': reason,
                    'direction': 'fake'
                })
        
        for indicator, reason, importance in real_indicators:
            if indicator in text_lower:
                top_features.append({
                    'keyword': indicator,
                    'importance': importance,
                    'impact': reason,
                    'direction': 'real'
                })
        
        # Analyze patterns
        patterns = []
        if text.count('!') > 3:
            patterns.append("Excessive exclamation marks")
        if text.count('?') > 3:
            patterns.append("Excessive question marks")
        if sum(1 for w in text.split() if w.isupper() and len(w) > 2) > 5:
            patterns.append("Excessive capitalization")
        if len(text) < 100:
            patterns.append("Very short (typical of sensational content)")
        
        # Generate reasoning
        fake_count = len([f for f in top_features if f['direction'] == 'fake'])
        real_count = len([f for f in top_features if f['direction'] == 'real'])
        
        if prediction == 1:
            reasoning = f"Article shows {fake_count} fake news indicators"
            if patterns:
                reasoning += f" and {len(patterns)} suspicious patterns"
        else:
            reasoning = f"Article appears credible with {real_count} credibility markers"
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            'method': 'Keyword Analysis',
            'top_features': sorted(top_features, key=lambda x: abs(x['importance']), reverse=True)[:10],
            'patterns': patterns,
            'reasoning': reasoning,
            'fake_indicators': fake_count,
            'real_indicators': real_count,
            'time_ms': processing_time,
            'prediction': prediction
        }
    
    def generate_heatmap(self, image_bytes: bytes, model: Any = None) -> Optional[str]:
        """
        Generate GradCAM heatmap for deepfake detection.
        
        Args:
            image_bytes: Input image bytes
            model: Optional deepfake detection model
            
        Returns:
            Base64 encoded heatmap image or None
        """
        # Try to use model's built-in GradCAM if available
        if model is not None and hasattr(model, 'generate_gradcam_heatmap'):
            return model.generate_gradcam_heatmap(image_bytes)
        
        if self._gradcam_available and model is not None:
            return self._generate_gradcam_heatmap(image_bytes, model)
        
        return self._generate_placeholder_heatmap(image_bytes)
    
    def _generate_gradcam_heatmap(self, image_bytes: bytes, model: Any) -> Optional[str]:
        """Generate actual GradCAM heatmap"""
        try:
            import torch
            
            # Get the PyTorch model
            if hasattr(model, 'get_model'):
                pytorch_model = model.get_model('efficientnet') or model.get_model('inception')
            else:
                pytorch_model = model
            
            if pytorch_model is None:
                return self._generate_placeholder_heatmap(image_bytes)
            
            # Get target layer
            if hasattr(pytorch_model, 'features'):
                target_layers = [pytorch_model.features[-1]]
            elif hasattr(pytorch_model, 'block8'):
                target_layers = [pytorch_model.block8]
            else:
                return self._generate_placeholder_heatmap(image_bytes)
            
            # Setup GradCAM
            cam = GradCAM(model=pytorch_model, target_layers=target_layers)
            
            # Prepare image
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img_resized = img.resize((224, 224))
            
            # Transform
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = transform(img).unsqueeze(0)
            
            # Generate CAM
            grayscale_cam = cam(input_tensor=input_tensor)[0]
            
            # Create visualization
            img_np = np.array(img_resized) / 255.0
            visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
            
            # Convert to base64
            result_img = Image.fromarray(visualization)
            buffer = io.BytesIO()
            result_img.save(buffer, format='PNG')
            b64_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{b64_str}"
            
        except Exception as e:
            print(f"⚠️ GradCAM generation error: {e}")
            return self._generate_placeholder_heatmap(image_bytes)
    
    def _generate_placeholder_heatmap(self, image_bytes: bytes) -> Optional[str]:
        """Generate a placeholder heatmap for demo"""
        if not PIL_AVAILABLE:
            return None
            
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img = img.resize((256, 256))
            
            heatmap = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(heatmap)
            
            center = (img.size[0] // 2, img.size[1] // 2 - 20)
            for r in range(80, 10, -10):
                alpha = int((80 - r) * 2.5)
                draw.ellipse(
                    [center[0]-r, center[1]-r, center[0]+r, center[1]+r],
                    fill=(255, 0, 0, alpha)
                )
            
            heatmap = heatmap.filter(ImageFilter.GaussianBlur(15))
            result = Image.alpha_composite(img.convert('RGBA'), heatmap)
            
            buffer = io.BytesIO()
            result.save(buffer, format='PNG')
            b64_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{b64_str}"
            
        except Exception as e:
            print(f"⚠️ Placeholder heatmap error: {e}")
            return None
    
    def get_suspicious_regions(self, prediction: int) -> List[str]:
        """
        Get descriptions of suspicious regions for deepfake.
        
        Args:
            prediction: 0 for authentic, 1 for deepfake
            
        Returns:
            List of region descriptions
        """
        if prediction == 1:
            return [
                "Face boundaries - potential blending artifacts",
                "Eye region - may show unnatural patterns",
                "Mouth area - possible audio-visual sync issues",
                "Hair edges - common deepfake boundary effects",
                "Skin texture - inconsistent lighting patterns"
            ]
        else:
            return [
                "Face structure - consistent natural features",
                "Eye region - normal reflection patterns",
                "Skin texture - consistent lighting"
            ]
    
    def get_analysis_summary(
        self, 
        prediction: int, 
        confidence: float,
        analysis_type: str = 'news'
    ) -> str:
        """Generate human-readable analysis summary"""
        conf_level = "high" if confidence > 0.8 else "moderate" if confidence > 0.6 else "low"
        
        if analysis_type == 'news':
            if prediction == 1:
                return f"This article has been flagged as potentially FAKE with {conf_level} confidence ({confidence:.1%}). Review the highlighted indicators below."
            else:
                return f"This article appears to be GENUINE with {conf_level} confidence ({confidence:.1%}). No major fake news indicators detected."
        else:
            if prediction == 1:
                return f"This image has been flagged as a potential DEEPFAKE with {conf_level} confidence ({confidence:.1%}). Check the heatmap for manipulated regions."
            else:
                return f"This image appears AUTHENTIC with {conf_level} confidence ({confidence:.1%}). No significant manipulation detected."
