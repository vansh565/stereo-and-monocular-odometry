import cv2
import numpy as np
import json
import os
from datetime import datetime

class VisualOdometry:
    def __init__(self):
        # ORB Feature Detector
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # MONOCULAR MODE
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None
        
        # STEREO MODE
        self.prev_left_gray = None
        self.prev_left_kp = None
        self.prev_left_des = None
        
        # Position tracking
        self.x = 0.0
        self.z = 0.0
        self.trajectory = []
        self.frame_count = 0
        self.current_mode = 'mono'
        
        # Camera parameters
        self.focal_length = 700.0
        self.baseline = 0.1
        
        # Movement smoothing
        self.movement_history_x = []
        self.movement_history_z = []
        
        # Create trajectories directory
        self.trajectories_dir = 'trajectories'
        os.makedirs(self.trajectories_dir, exist_ok=True)
        
        print("Visual Odometry System Initialized")
        print("Modes: MONOCULAR (1 camera) | STEREO (2 cameras required)")
    
    def reset(self):
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None
        self.prev_left_gray = None
        self.prev_left_kp = None
        self.prev_left_des = None
        self.x = 0.0
        self.z = 0.0
        self.frame_count = 0
        self.trajectory = []
        self.movement_history_x = []
        self.movement_history_z = []
        print("System Reset - Position Zeroed")
    
    def save_trajectory(self):
        if len(self.trajectory) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.trajectories_dir}/trajectory_{self.current_mode}_{timestamp}.json"
            data = {
                'mode': self.current_mode,
                'timestamp': timestamp,
                'frames': self.frame_count,
                'trajectory': self.trajectory,
                'final_position': [float(self.x), float(self.z)]
            }
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Trajectory saved: {filename}")
            return filename
        return None
    
    def calculate_optical_flow(self, prev_gray, gray):
        try:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            avg_flow_x = float(np.mean(flow[:,:,0]))
            avg_flow_y = float(np.mean(flow[:,:,1]))
            move_x = avg_flow_x * 0.8 if abs(avg_flow_x) > 0.2 else 0.0
            move_z = avg_flow_y * 0.8 if abs(avg_flow_y) > 0.2 else 0.0
            return move_x, move_z
        except:
            return 0.0, 0.0
    
    def calculate_feature_movement(self, prev_kp, prev_des, kp, des):
        try:
            matches = self.bf.match(prev_des, des)
            if len(matches) > 10:
                matches = sorted(matches, key=lambda x: x.distance)[:50]
                move_sum_x = 0.0
                move_sum_z = 0.0
                valid_matches = 0
                for match in matches:
                    if match.queryIdx < len(prev_kp) and match.trainIdx < len(kp):
                        px1, py1 = prev_kp[match.queryIdx].pt
                        px2, py2 = kp[match.trainIdx].pt
                        move_sum_x += (px2 - px1)
                        move_sum_z += (py2 - py1)
                        valid_matches += 1
                if valid_matches > 0:
                    return (move_sum_x / valid_matches) * 0.3, (move_sum_z / valid_matches) * 0.3
        except:
            pass
        return 0.0, 0.0
    
    def apply_movement(self, move_x, move_z):
        self.movement_history_x.append(move_x)
        self.movement_history_z.append(move_z)
        if len(self.movement_history_x) > 5:
            self.movement_history_x.pop(0)
            self.movement_history_z.pop(0)
        if len(self.movement_history_x) > 0:
            avg_move_x = sum(self.movement_history_x) / len(self.movement_history_x)
            avg_move_z = sum(self.movement_history_z) / len(self.movement_history_z)
            self.x += avg_move_x * 1.2
            self.z += avg_move_z * 1.2
    
    def process_monocular(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        feature_count = len(kp) if kp else 0
        move_x, move_z = 0.0, 0.0
        
        if self.prev_gray is not None:
            move_x, move_z = self.calculate_optical_flow(self.prev_gray, gray)
            if move_x == 0 and move_z == 0 and self.prev_des is not None and des is not None:
                move_x, move_z = self.calculate_feature_movement(self.prev_kp, self.prev_des, kp, des)
        
        self.apply_movement(move_x, move_z)
        self.frame_count += 1
        self.current_mode = 'mono'
        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_des = des
        
        self.trajectory.append({'frame': self.frame_count, 'x': float(self.x), 'z': float(self.z), 'mode': 'mono'})
        if len(self.trajectory) > 500:
            self.trajectory.pop(0)
        
        if self.frame_count % 30 == 0:
            print(f"MONO - Frame {self.frame_count}: X={self.x:.2f}, Z={self.z:.2f}, Features={feature_count}")
        
        return float(self.x), float(self.z), int(feature_count)
    
    def process_stereo(self, left_frame, right_frame):
        """Process stereo frames - requires two actual cameras"""
        gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        kp_left, des_left = self.orb.detectAndCompute(gray_left, None)
        feature_count = len(kp_left) if kp_left else 0
        move_x, move_z = 0.0, 0.0
        
        # Calculate depth from stereo disparity
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(gray_left, gray_right)
        valid_disparities = disparity[disparity > 0]
        depth = 0.0
        if len(valid_disparities) > 0:
            avg_disparity = np.mean(valid_disparities)
            if avg_disparity > 0:
                depth = (self.focal_length * self.baseline) / avg_disparity
                depth = float(min(10.0, max(0.5, depth)))
        
        if self.prev_left_gray is not None:
            move_x, move_z = self.calculate_optical_flow(self.prev_left_gray, gray_left)
            if move_x == 0 and move_z == 0 and self.prev_left_des is not None and des_left is not None:
                move_x, move_z = self.calculate_feature_movement(self.prev_left_kp, self.prev_left_des, kp_left, des_left)
        
        if depth > 0:
            depth_scale = min(1.5, max(0.5, 2.0 / depth))
            move_x *= depth_scale
            move_z *= depth_scale
        
        self.apply_movement(move_x, move_z)
        self.frame_count += 1
        self.current_mode = 'stereo'
        self.prev_left_gray = gray_left
        self.prev_left_kp = kp_left
        self.prev_left_des = des_left
        
        self.trajectory.append({'frame': self.frame_count, 'x': float(self.x), 'z': float(self.z), 'mode': 'stereo', 'depth': depth})
        if len(self.trajectory) > 500:
            self.trajectory.pop(0)
        
        if self.frame_count % 30 == 0:
            print(f"STEREO - Frame {self.frame_count}: X={self.x:.2f}, Z={self.z:.2f}, Depth={depth:.2f}")
        
        return float(self.x), float(self.z), int(feature_count)