from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename  # Import secure_filename function
import os

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_video():
    uploaded_file = request.files['video']
    if uploaded_file:
        filename = secure_filename(uploaded_file.filename)
        video_path = os.path.join('vidFolder', filename)
        uploaded_file.save(video_path)
        # Process the video as needed
        return jsonify({'filename': filename})
    return jsonify({'error': 'No file uploaded'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)