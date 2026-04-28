from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64
import time
import logging
import os

# For production
import sys
import platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

# For production on Linux servers (like Render, AWS, GCP)
# OpenCV needs special configuration on headless servers
if platform.system() == 'Linux':
    import cv2
    # Use headless OpenCV
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

from vo import VisualOdometry

vo = VisualOdometry()

# Get port from environment variable for cloud deployment
PORT = int(os.environ.get('PORT', 5000))

def decode_image(base64_str):
    try:
        if ',' in base64_str:
            base64_str = base64_str.split(",")[1]
        
        missing_padding = len(base64_str) % 4
        if missing_padding:
            base64_str += '=' * (4 - missing_padding)
        
        img_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Decode error: {e}")
        raise

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('frontend', filename)

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route("/process", methods=["POST"])
def process():
    try:
        start_time = time.time()
        data = request.json
        frame = decode_image(data["image"])
        mode = data.get('mode', 'mono')
        
        if mode == 'stereo' and 'right_image' in data and data['right_image']:
            right_frame = decode_image(data['right_image'])
            x, z, features = vo.process_stereo(frame, right_frame)
        else:
            x, z, features = vo.process_monocular(frame)
        
        response_data = {
            "x": float(x),
            "z": float(z),
            "feature_count": int(features),
            "frame_count": int(vo.frame_count),
            "mode": mode,
            "processing_time": float((time.time() - start_time) * 1000),
            "success": True
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e), "success": False}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "timestamp": time.time()})

@app.route("/reset", methods=["POST"])
def reset():
    vo.reset()
    return jsonify({"success": True})

@app.route("/save_trajectory", methods=["POST"])
def save_trajectory():
    filename = vo.save_trajectory()
    if filename:
        return jsonify({"success": True, "filename": filename})
    else:
        return jsonify({"success": False, "message": "No trajectory data"})

if __name__ == "__main__":
    print("=" * 60)
    print("VISUAL ODOMETRY SYSTEM - PRODUCTION MODE")
    print("=" * 60)
    print(f"Server running on port {PORT}")
    print("=" * 60)
    
    # Use production server
    if os.environ.get('ENV') == 'production':
        from waitress import serve
        serve(app, host='0.0.0.0', port=PORT)
    else:
        app.run(debug=False, host='0.0.0.0', port=PORT)
