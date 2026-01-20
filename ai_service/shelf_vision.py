from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import os

# Import enhanced image analysis
try:
    from image_analysis import image_analyzer
    ENHANCED_DETECTION = True
except ImportError:
    ENHANCED_DETECTION = False
    print("Warning: Enhanced detection not available")

class ShelfDetector:
    def __init__(self):
        self.model = None
        try:
            # Try to load custom trained retail model first
            custom_model_path = 'models/retail_yolov8s_best.pt'
            if os.path.exists(custom_model_path):
                self.model = YOLO(custom_model_path)
                print(f"✓ Custom retail model loaded from {custom_model_path}")
                print(f"  - Trained on {len(self.model.names)} product classes")
                self.is_custom_model = True
            else:
                # Fallback to generic YOLOv8
                self.model = YOLO('yolov8n.pt')
                print("✓ Generic YOLOv8 model loaded (custom model not found)")
                print("  - For better accuracy, train custom model (see training_quickstart.md)")
                self.is_custom_model = False
        except Exception as e:
            print(f"Warning: YOLO model failed to load. Using Mock Vision. Error: {e}")
            self.is_custom_model = False
        
        # Indian retail product categories
        self.retail_categories = {
            'saree': 'Saree',
            'fabric': 'Fabric/Cloth',
            'fabric_roll': 'Fabric Roll',
            'dress_material': 'Dress Material',
            'clothing': 'Clothing Item',
            'dupatta': 'Dupatta/Scarf',
            'kurta': 'Kurta/Kurti',
            'shirt': 'Shirt',
            'pant': 'Pant/Trouser'
        }

    def analyze_image(self, image_bytes):
        if not self.model:
            return self.mock_response()

        try:
            # Convert bytes to image
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            
            # Run inference
            results = self.model(image)
            
            # Process results with enhanced detection
            detected_items = {}
            total_count = 0
            boxes = []
            
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    generic_name = self.model.names[cls]
                    box_coords = box.xyxy[0].tolist()
                    
                    # Higher confidence threshold for custom model to reduce false positives
                    min_confidence = 0.4 if self.is_custom_model else 0.25
                    
                    if conf > min_confidence:
                        # Enhanced classification for retail products
                        if ENHANCED_DETECTION and not self.is_custom_model:
                            corrected_name, new_conf, explanation = image_analyzer.classify_retail_product(
                                box_coords,
                                image_np,
                                generic_name,
                                conf
                            )
                            
                            # Use corrected classification
                            final_name = corrected_name
                            final_conf = new_conf
                            
                            # Map to retail category if available
                            if final_name in self.retail_categories:
                                display_name = self.retail_categories[final_name]
                            else:
                                display_name = final_name.replace('_', ' ').title()
                        else:
                            # For custom model, use direct classification
                            final_name = generic_name
                            display_name = generic_name.replace('_', ' ').title()
                            final_conf = conf
                        
                        # Advanced filtering: Remove unlikely detections
                        if self.is_custom_model:
                            skip_detection = False
                            
                            # Define similar/related categories
                            sweet_categories = ['chocolate', 'candy', 'dessert', 'gum']
                            hygiene_categories = ['personal_hygiene', 'tissue']
                            
                            # If we have many sweet items, be very strict with hygiene detections
                            if final_name in hygiene_categories:
                                sweet_count = sum(detected_items.get(cat.title(), 0) for cat in sweet_categories)
                                
                                # Require very high confidence for hygiene if we have lots of sweets
                                if sweet_count > 10 and conf < 0.6:
                                    skip_detection = True
                                elif sweet_count > 5 and conf < 0.5:
                                    skip_detection = True
                            
                            # Skip if flagged
                            if skip_detection:
                                continue
                        
                        # Count items
                        detected_items[display_name] = detected_items.get(display_name, 0) + 1
                        total_count += 1
                        boxes.append(box_coords)
            
            # Post-processing: Merge similar categories and clean up
            if self.is_custom_model:
                # Merge candy and chocolate if both present (they're often confused)
                if 'Candy' in detected_items and 'Chocolate' in detected_items:
                    # If one is much larger, it's probably the correct one
                    candy_count = detected_items['Candy']
                    choc_count = detected_items['Chocolate']
                    
                    if choc_count > candy_count * 3:
                        # Mostly chocolate, merge candy into chocolate
                        detected_items['Chocolate'] += candy_count
                        del detected_items['Candy']
                    elif candy_count > choc_count * 3:
                        # Mostly candy, merge chocolate into candy
                        detected_items['Candy'] += choc_count
                        del detected_items['Chocolate']
                
                # Remove very small counts if we have a dominant category
                total_items = sum(detected_items.values())
                if total_items > 20:
                    # Find dominant category
                    max_count = max(detected_items.values())
                    if max_count > total_items * 0.7:  # One category is >70%
                        # Remove items with count < 5% of total
                        min_threshold = max(2, int(total_items * 0.05))
                        detected_items = {k: v for k, v in detected_items.items() if v >= min_threshold}
                
                # Recalculate total
                total_count = sum(detected_items.values())

            # Calculate shelf health
            gap_score = self.calculate_gap_score(boxes, image.size)
            
            # Detect stacking pattern (common for sarees/fabrics)
            is_stacked = False
            if ENHANCED_DETECTION and len(boxes) > 2:
                is_stacked = image_analyzer.is_stacked_pattern(boxes, image.size[1])
            
            # Build response
            model_type = "Custom Retail Model" if self.is_custom_model else "Generic YOLO"
            if ENHANCED_DETECTION and not self.is_custom_model:
                model_type = "Enhanced Retail Detection (Generic)"
            
            response = {
                "item_counts": detected_items,
                "total_items": total_count,
                "shelf_health": "Good" if gap_score > 70 else "Needs Restock",
                "gap_score": gap_score,
                "anomalies": [],
                "detection_mode": model_type
            }
            
            # Add insights
            if is_stacked:
                response["anomalies"].append("Stacked items detected (likely folded fabrics/sarees)")
            
            if total_count == 0:
                response["anomalies"].append("No items detected - shelf may be empty")
            
            return response

        except Exception as e:
            print(f"Error during vision analysis: {e}")
            import traceback
            traceback.print_exc()
            return self.mock_response()

    def calculate_gap_score(self, boxes, img_size):
        """
        Calculate shelf coverage score
        """
        if not boxes:
            return 0
        
        # Calculate total area covered by boxes
        total_area = 0
        img_area = img_size[0] * img_size[1]
        
        for box in boxes:
            width = box[2] - box[0]
            height = box[3] - box[1]
            total_area += width * height
        
        # Coverage percentage
        coverage = (total_area / img_area) * 100
        
        # Score based on coverage (good shelf should be 60-90% covered)
        if coverage > 90:
            return 85  # Too crowded
        elif coverage > 60:
            return 95  # Good coverage
        elif coverage > 30:
            return 70  # Moderate
        else:
            return 50  # Needs restock

    def mock_response(self):
        return {
            "item_counts": {"Saree": 9, "Fabric": 3},
            "total_items": 12,
            "shelf_health": "Good (Mock)",
            "gap_score": 85,
            "anomalies": [],
            "detection_mode": "Mock (Model not loaded)"
        }

detector = ShelfDetector()
