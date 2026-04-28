import cv2
import numpy as np
import random

class YOLODetector:
    def __init__(self):
        self.use_yolo = False
        print("YOLO Detector ready (simulation mode)")
    
    def detect_objects(self, frame):
        """Detect objects in frame - returns serializable data"""
        detected_objects = []
        
        # Simulated detection with pure Python types
        h, w = frame.shape[:2]
        
        # Random detection for demo (convert to Python ints/floats)
        if random.random() > 0.7:
            obj_count = random.randint(1, 3)
            classes = ['person', 'car', 'bicycle', 'dog', 'cat', 'chair', 'bottle', 'laptop']
            
            for i in range(obj_count):
                detected_objects.append({
                    'class': random.choice(classes),
                    'confidence': float(random.uniform(0.6, 0.95)),
                    'bbox': [
                        int(random.randint(50, w - 150)),
                        int(random.randint(50, h - 150)),
                        int(random.randint(50, 150)),
                        int(random.randint(80, 200))
                    ]
                })
        
        return detected_objects
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes on frame"""
        for det in detections:
            x, y, w, h = det['bbox']
            label = f"{det['class']} ({det['confidence']:.2f})"
            
            # Use Python ints for drawing
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w, y), color, -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame