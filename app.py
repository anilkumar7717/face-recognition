from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import logging
import insightface
from typing import List
from insightface.app import FaceAnalysis
import cv2

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

def preprocess_image(image_file, max_size=(800, 800)):
    img = Image.open(image_file).convert('RGB')
    img = ImageOps.exif_transpose(img)
    if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return np.array(img)

def get_best_face_embedding(img_array):
    try:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        faces = face_app.get(img_bgr)

        if not faces:
            return None

        largest_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        return largest_face.embedding.tolist()

    except Exception as e:
        logger.error(f"Error in face embedding: {e}")
        return None

@app.route('/convertToEncoding', methods=['POST'])
def convert_to_encoding():
    try:
        if 'image' not in request.files:
            resp = {'error': 'No image file provided'}
            logger.info(f"Returning: {resp}")
            return jsonify(resp), 400

        image_file = request.files['image']
        if image_file.filename == '':
            resp = {'error': 'No selected file'}
            logger.info(f"Returning: {resp}")
            return jsonify(resp), 400

        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        file_ext = image_file.filename.rsplit('.', 1)[1].lower() if '.' in image_file.filename else ''
        if file_ext not in allowed_extensions:
            resp = {'error': 'Invalid file type'}
            logger.info(f"Returning: {resp}")
            return jsonify(resp), 400

        img_array = preprocess_image(image_file)
        face_embedding = get_best_face_embedding(img_array)

        if face_embedding is None:
            resp = {'error': 'No face detected or face too unclear'}
            logger.info(f"Returning: {resp}")
            return jsonify(resp), 400

        resp = {
            'success': True,
            'face_encoding': face_embedding,
            'encoding_quality': 'good'
        }
        logger.info(f"Returning: {resp}")
        return jsonify(resp)

    except Exception as e:
        logger.error(f"Error converting image to face encoding: {str(e)}")
        return jsonify({'error': 'Failed to process image'}), 500

@app.route('/faceDetection', methods=['POST'])
def verify_face():
    try:
        data = request.json
        if not data:
            logger.warning("No JSON data received")
            resp = {'error': 'No JSON data provided'}
            logger.info(f"Returning: {resp}")
            return jsonify(resp), 400

        captured_encoding = data.get('capturedEncoding')
        enrolled_encodings = data.get('enrolledEncodings')

        if not captured_encoding or not enrolled_encodings:
            resp = {'error': 'Missing capturedEncoding or enrolledEncodings'}
            logger.info(f"Returning: {resp}")
            return jsonify(resp), 400

        captured_encoding = np.array(captured_encoding, dtype=np.float32)

        if captured_encoding.shape[0] != 512:
            resp = {'error': 'Invalid face encoding format'}
            logger.info(f"Returning: {resp}")
            return jsonify(resp), 400

        best_match = None
        best_distance = float('inf')

        for index, enrolled in enumerate(enrolled_encodings):
            try:
                enrolled_encoding = np.array(enrolled['encoding'], dtype=np.float32)

                if enrolled_encoding.shape[0] != 512:
                    logger.warning(f"Skipping invalid enrolled encoding at index {index}")
                    continue

                dot_product = np.dot(captured_encoding, enrolled_encoding)
                norm_product = np.linalg.norm(captured_encoding) * np.linalg.norm(enrolled_encoding)
                similarity = dot_product / norm_product
                distance = 1 - similarity

                logger.info(f"Encoding {index + 1} (ID={enrolled.get('id', index)}): Distance = {distance:.4f}")

                if distance < best_distance:
                    best_distance = distance
                    best_match = {
                        'id': enrolled.get('id', index),
                        'distance': float(distance)
                    }

            except Exception as e:
                logger.error(f"Error comparing encoding at index {index}: {e}")
                continue

        if best_match:
            distance_value = float(best_distance)
            confidence = 1.0 - distance_value
            match_percentage = round(confidence * 100, 2)

            if confidence >= 0.85: 
                resp = {
                    'match': True,
                    'matchedEncodingId': best_match['id'],
                    'distance': distance_value,
                    'confidence': round(confidence, 3),
                    'match_percentage': match_percentage
                } 
                logger.info(f" MATCH FOUND: {resp}")
                return jsonify(resp)
            else:
                resp = {
                    'match': False,
                    'message': 'Face found, but confidence is below 85%',
                    'min_distance': distance_value,
                    'confidence': round(confidence, 3),
                    'match_percentage': match_percentage
                }
                logger.info(f"MATCH TOO WEAK: {resp}")
                return jsonify(resp)
        else:
            resp = {
                'match': False,
                'message': 'No matching face found',
                'min_distance': None
            }
            logger.info(f"NO MATCH FOUND: {resp}")
            return jsonify(resp)

    except Exception as e:
        logger.error(f"Exception during face verification: {str(e)}")
        return jsonify({'error': 'Face verification failed'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'face-recognition-api-insightface'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
