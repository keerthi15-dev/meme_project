"""
Image Analysis Utilities for Retail Product Detection
Provides color, texture, and pattern analysis for better classification
"""

import cv2
import numpy as np
from PIL import Image
import io


class ImageAnalyzer:
    """
    Analyze image characteristics for retail product classification
    """
    
    def __init__(self):
        self.fabric_keywords = ['saree', 'fabric', 'cloth', 'textile', 'dress material']
    
    def get_dominant_colors(self, image_region, k=3):
        """
        Extract dominant colors from image region
        
        Returns:
            List of (color, percentage) tuples
        """
        try:
            # Convert to RGB if needed
            if isinstance(image_region, np.ndarray):
                if len(image_region.shape) == 2:  # Grayscale
                    image_region = cv2.cvtColor(image_region, cv2.COLOR_GRAY2RGB)
                elif image_region.shape[2] == 4:  # RGBA
                    image_region = cv2.cvtColor(image_region, cv2.COLOR_RGBA2RGB)
            
            # Reshape to list of pixels
            pixels = image_region.reshape(-1, 3).astype(np.float32)
            
            # K-means clustering to find dominant colors
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Count pixels per cluster
            unique, counts = np.unique(labels, return_counts=True)
            
            # Calculate percentages
            total_pixels = len(pixels)
            colors = []
            for i, center in enumerate(centers):
                percentage = counts[i] / total_pixels
                colors.append((center.astype(int), percentage))
            
            return sorted(colors, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            print(f"Color analysis error: {e}")
            return []
    
    def is_colorful(self, colors, threshold=0.3):
        """
        Check if image has vibrant, multiple colors (characteristic of sarees/fabrics)
        
        Args:
            colors: List of (color, percentage) from get_dominant_colors
            threshold: Minimum percentage for a color to be significant
        """
        if not colors or len(colors) < 2:
            return False
        
        # Count significant colors
        significant_colors = [c for c in colors if c[1] > threshold]
        
        # Check color variance (saturation)
        saturations = []
        for color, _ in colors[:3]:  # Top 3 colors
            # Convert RGB to HSV to get saturation
            rgb = np.uint8([[color]])
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            saturation = hsv[0][0][1]
            saturations.append(saturation)
        
        avg_saturation = np.mean(saturations)
        
        # Colorful if: multiple colors AND high saturation
        return len(significant_colors) >= 2 and avg_saturation > 80
    
    def analyze_texture(self, image_region):
        """
        Analyze texture to distinguish fabric from paper/books
        
        Returns:
            dict with texture features
        """
        try:
            # Convert to grayscale
            if len(image_region.shape) == 3:
                gray = cv2.cvtColor(image_region, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_region
            
            # Calculate texture features
            
            # 1. Edge density (books have sharp edges, fabrics are softer)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # 2. Variance (fabrics have more texture variation)
            variance = np.var(gray)
            
            # 3. Gradient magnitude (fabrics have smoother gradients)
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(gx**2 + gy**2).mean()
            
            return {
                'edge_density': edge_density,
                'variance': variance,
                'gradient_magnitude': gradient_mag,
                'is_fabric_like': edge_density < 0.15 and variance > 500
            }
            
        except Exception as e:
            print(f"Texture analysis error: {e}")
            return {'is_fabric_like': False}
    
    def get_aspect_ratio(self, box):
        """
        Calculate aspect ratio of bounding box
        
        Args:
            box: [x1, y1, x2, y2]
        """
        width = box[2] - box[0]
        height = box[3] - box[1]
        
        if height == 0:
            return 1.0
        
        return width / height
    
    def is_stacked_pattern(self, boxes, image_height):
        """
        Detect if objects are stacked (common for folded sarees/fabrics)
        
        Args:
            boxes: List of bounding boxes
            image_height: Height of full image
        """
        if len(boxes) < 2:
            return False
        
        # Sort boxes by y-coordinate
        sorted_boxes = sorted(boxes, key=lambda b: b[1])
        
        # Check vertical spacing
        vertical_gaps = []
        for i in range(len(sorted_boxes) - 1):
            gap = sorted_boxes[i+1][1] - sorted_boxes[i][3]
            vertical_gaps.append(gap)
        
        # Stacked if gaps are small and consistent
        if vertical_gaps:
            avg_gap = np.mean(vertical_gaps)
            gap_consistency = np.std(vertical_gaps)
            
            # Small, consistent gaps indicate stacking
            return avg_gap < (image_height * 0.05) and gap_consistency < 20
        
        return False
    
    def classify_retail_product(self, box, image, generic_class, confidence):
        """
        Enhanced classification for retail products
        
        Args:
            box: Bounding box [x1, y1, x2, y2]
            image: Full image (PIL or numpy)
            generic_class: Class from generic YOLO
            confidence: Detection confidence
        
        Returns:
            (corrected_class, confidence, explanation)
        """
        try:
            # Convert PIL to numpy if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Crop region
            x1, y1, x2, y2 = [int(c) for c in box]
            region = image[y1:y2, x1:x2]
            
            if region.size == 0:
                return generic_class, confidence, "No region"
            
            # Analyze characteristics
            colors = self.get_dominant_colors(region)
            texture = self.analyze_texture(region)
            aspect_ratio = self.get_aspect_ratio(box)
            is_colorful = self.is_colorful(colors)
            
            # Retail-specific classification rules
            
            # Rule 1: Books misclassified as sarees/fabrics
            if generic_class == "book":
                if is_colorful and texture['is_fabric_like']:
                    # Likely a saree or fabric
                    if aspect_ratio > 1.5:
                        return "fabric_roll", 0.75, "Colorful, fabric texture, horizontal"
                    else:
                        return "saree", 0.80, "Colorful, fabric texture, folded"
                elif is_colorful:
                    return "dress_material", 0.70, "Colorful but uncertain texture"
            
            # Rule 2: Bottles/containers might be fabric rolls
            if generic_class in ["bottle", "cup", "vase"]:
                if aspect_ratio > 2 and texture['is_fabric_like']:
                    return "fabric_roll", 0.65, "Cylindrical fabric roll"
            
            # Rule 3: Backpack/handbag might be clothing
            if generic_class in ["backpack", "handbag", "suitcase"]:
                if is_colorful and texture['is_fabric_like']:
                    return "clothing", 0.70, "Fabric-like, colorful"
            
            # Rule 4: Keep high-confidence generic detections
            if confidence > 0.7 and generic_class not in ["book"]:
                return generic_class, confidence, "High confidence generic"
            
            # Default: return generic class
            return generic_class, confidence, "Generic detection"
            
        except Exception as e:
            print(f"Classification error: {e}")
            return generic_class, confidence, f"Error: {e}"


# Singleton instance
image_analyzer = ImageAnalyzer()
