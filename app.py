from flask import Flask, request, jsonify, send_file, render_template
import os
from werkzeug.utils import secure_filename
import cv2
import torch
# Set upload and output folder paths
app = Flask(__name__)

# Set upload and output folder paths
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load pre-trained YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Perform object detection
    try:
        results = model(filepath)
        results.render()  # Render results (bounding boxes and labels)

        # Save the updated image using Ultralytics default path
        results.save()  # This saves to runs/detect/exp*, not predict
        # Find the latest exp* directory
        import glob
        import shutil
        detect_dir = os.path.join('runs', 'detect')
        exp_dirs = [d for d in glob.glob(os.path.join(detect_dir, 'exp*')) if os.path.isdir(d)]
        if not exp_dirs:
            raise Exception('No exp* directory found in runs/detect.')
        latest_exp = max(exp_dirs, key=os.path.getctime)
        # Find the image file in the latest exp* directory
        list_of_files = glob.glob(os.path.join(latest_exp, '*'))
        if not list_of_files:
            raise Exception('No output file found in latest exp* directory.')
        latest_file = max(list_of_files, key=os.path.getctime)
        # Copy to output folder with original filename
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        shutil.copy(latest_file, output_path)
        return send_file(output_path, mimetype='image/jpeg')
    except Exception as e:
        print(f"Error during detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
