"""
AI Customer Behavior Heatmaps
Tracks customer movements and generates visual heatmaps for retail optimization
Based on 2024 research: $1.59B market → $3.63B by 2029
"""

import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import io
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import base64

class CustomerHeatmapAnalyzer:
    def __init__(self):
        self.model = None
        try:
            # Load YOLOv8 for person detection
            self.model = YOLO('yolov8n.pt')
            print("✓ YOLOv8 loaded for person detection")
        except Exception as e:
            print(f"Warning: YOLO model failed to load: {e}")
    
    def detect_people(self, image_np):
        """
        Detect people in image using YOLOv8
        Returns list of person positions (x, y)
        """
        if not self.model:
            return []
        
        # Run detection
        results = self.model(image_np, classes=[0])  # Class 0 = person
        
        positions = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > 0.3:  # Confidence threshold
                    # Get center of bounding box
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    positions.append((center_x, center_y))
        
        return positions
    
    def generate_heatmap(self, positions, image_shape, sigma=50):
        """
        Generate heatmap from person positions
        """
        height, width = image_shape[:2]
        
        # Create empty heatmap
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Add gaussian blobs at each position
        for x, y in positions:
            if 0 <= x < width and 0 <= y < height:
                heatmap[y, x] += 1
        
        # Apply gaussian filter for smooth heatmap
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def create_heatmap_visualization(self, image_np, heatmap):
        """
        Create visual heatmap overlay on original image
        """
        # Create custom colormap (blue -> green -> yellow -> red)
        colors = ['#000033', '#0000FF', '#00FF00', '#FFFF00', '#FF0000']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Original image with overlay
        ax1.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        im1 = ax1.imshow(heatmap, cmap=cmap, alpha=0.6, interpolation='bilinear')
        ax1.set_title('Customer Heatmap Overlay', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Heatmap only
        im2 = ax2.imshow(heatmap, cmap=cmap, interpolation='bilinear')
        ax2.set_title('Heatmap (High Activity = Red)', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Activity Level', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_base64
    
    def analyze_zones(self, heatmap, image_shape):
        """
        Divide image into zones and analyze activity
        """
        height, width = image_shape[:2]
        
        # Divide into 3x3 grid
        zones = []
        zone_height = height // 3
        zone_width = width // 3
        
        zone_names = [
            ['Top-Left', 'Top-Center', 'Top-Right'],
            ['Middle-Left', 'Center', 'Middle-Right'],
            ['Bottom-Left', 'Bottom-Center', 'Bottom-Right']
        ]
        
        for i in range(3):
            for j in range(3):
                y_start = i * zone_height
                y_end = (i + 1) * zone_height if i < 2 else height
                x_start = j * zone_width
                x_end = (j + 1) * zone_width if j < 2 else width
                
                zone_heatmap = heatmap[y_start:y_end, x_start:x_end]
                activity = float(zone_heatmap.mean())
                
                zones.append({
                    'name': zone_names[i][j],
                    'activity': activity,
                    'position': (i, j)
                })
        
        # Sort by activity
        zones.sort(key=lambda x: x['activity'], reverse=True)
        
        return zones
    
    def generate_recommendations(self, zones, total_people):
        """
        Generate AI recommendations based on heatmap analysis
        """
        recommendations = []
        
        if total_people == 0:
            return ["No customers detected in image. Upload store image with customers for analysis."]
        
        # Find hot and cold zones
        hot_zones = [z for z in zones if z['activity'] > 0.6]
        cold_zones = [z for z in zones if z['activity'] < 0.2]
        
        # Recommendations
        if hot_zones:
            hot_zone_names = ', '.join([z['name'] for z in hot_zones[:2]])
            recommendations.append(
                f"🔥 High traffic in {hot_zone_names}. Consider placing premium products here for +15% visibility."
            )
        
        if cold_zones:
            cold_zone_names = ', '.join([z['name'] for z in cold_zones[:2]])
            recommendations.append(
                f"❄️ Low traffic in {cold_zone_names}. Move promotional items here or improve signage."
            )
        
        # Center zone analysis
        center_zone = next((z for z in zones if z['name'] == 'Center'), None)
        if center_zone and center_zone['activity'] < 0.4:
            recommendations.append(
                "💡 Center zone underutilized. Place high-margin products in center for better visibility."
            )
        
        # Traffic distribution
        top_activity = zones[0]['activity']
        bottom_activity = zones[-1]['activity']
        if top_activity > bottom_activity * 3:
            recommendations.append(
                "⚖️ Uneven traffic distribution. Balance product placement across zones for +10% conversion."
            )
        
        if not recommendations:
            recommendations.append(
                "✅ Good traffic distribution. Monitor over time for optimization opportunities."
            )
        
        return recommendations
    
    def analyze_image(self, image_bytes):
        """
        Main analysis function
        """
        try:
            # Convert bytes to image
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            
            # Detect people
            positions = self.detect_people(image_np)
            
            # Generate heatmap
            heatmap = self.generate_heatmap(positions, image_np.shape)
            
            # Create visualization
            heatmap_img = self.create_heatmap_visualization(image_np, heatmap)
            
            # Analyze zones
            zones = self.analyze_zones(heatmap, image_np.shape)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(zones, len(positions))
            
            # Calculate metrics
            hot_zones = len([z for z in zones if z['activity'] > 0.6])
            cold_zones = len([z for z in zones if z['activity'] < 0.2])
            
            return {
                "total_customers": len(positions),
                "heatmap_image": heatmap_img,
                "zones": zones[:5],  # Top 5 zones
                "recommendations": recommendations,
                "metrics": {
                    "hot_zones": hot_zones,
                    "cold_zones": cold_zones,
                    "coverage_score": int((1 - (cold_zones / 9)) * 100)
                }
            }
        
        except Exception as e:
            print(f"Error in heatmap analysis: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "total_customers": 0,
                "recommendations": ["Error analyzing image. Please try again."]
            }

# Initialize analyzer
heatmap_analyzer = CustomerHeatmapAnalyzer()
