from flask import Flask, request, jsonify
import face_recognition
from flask_cors import CORS
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image(image_file, max_size=(800, 800)):
    """
    Preprocess image for better face detection
    """
    img = Image.open(image_file).convert('RGB')
    img = ImageOps.exif_transpose(img)
    
    # Resize if image is too large (improves performance)
    if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    return np.array(img)

def get_best_face_encoding(img_array, model='hog'):
    """
    Get the best face encoding from an image
    Uses multiple detection methods if needed
    """
    try:
        # Try HOG first (faster)
        face_locations = face_recognition.face_locations(img_array, model=model)
        if not face_locations:
            # Try CNN if HOG fails (more accurate but slower)
            face_locations = face_recognition.face_locations(img_array, model='cnn')
        
        if not face_locations:
            return None
        
        # Get encodings for all detected faces
        face_encodings = face_recognition.face_encodings(img_array, face_locations)
        
        if not face_encodings:
            return None
        
        # If multiple faces, return the largest one (most prominent)
        if len(face_encodings) > 1:
            largest_face_idx = max(range(len(face_locations)), 
                                 key=lambda i: (face_locations[i][2] - face_locations[i][0]) * 
                                             (face_locations[i][1] - face_locations[i][3]))
            return face_encodings[largest_face_idx]
        
        return face_encodings[0]
    
    except Exception as e:
        logger.error(f"Error in face encoding: {e}")
        return None

@app.route('/convertToEncoding', methods=['POST'])
def convert_to_encoding():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        file_ext = image_file.filename.rsplit('.', 1)[1].lower() if '.' in image_file.filename else ''
        if file_ext not in allowed_extensions:
            return jsonify({'error': 'Invalid file type'}), 400
        
        img_array = preprocess_image(image_file)
        face_encoding = get_best_face_encoding(img_array)
        
        if face_encoding is None:
            return jsonify({'error': 'No face detected or face too unclear'}), 400
        
        encoding_list = face_encoding.tolist()
        
        return jsonify({
            'success': True,
            'face_encoding': encoding_list,
            'encoding_quality': 'good'  # Could implement quality scoring
        })
    
    except Exception as e:
        logger.error(f"Error converting image to face encoding: {str(e)}")
        return jsonify({'error': 'Failed to process image'}), 500

@app.route('/faceDetection', methods=['POST'])
def verify_face():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        captured_encoding = data.get('capturedEncoding')
        enrolled_encodings = data.get('enrolledEncodings')
        custom_tolerance = data.get('tolerance', 0.6)  # Allow custom tolerance
        
        if not captured_encoding or not enrolled_encodings:
            return jsonify({
                'error': 'Missing capturedEncoding or enrolledEncodings'
            }), 400
        
        # Validate tolerance range
        if not (0.1 <= custom_tolerance <= 1.0):
            custom_tolerance = 0.6
        
        captured_face_encoding = np.array(captured_encoding)
        
        # Validate encoding dimensions
        if captured_face_encoding.shape[0] != 128:
            return jsonify({'error': 'Invalid face encoding format'}), 400
        
        best_match = None
        best_distance = float('inf')
        matches_found = []
        
        # Compare with all enrolled encodings
        for index, enrolled_encoding in enumerate(enrolled_encodings):
            try:
                enrolled_face_encoding = np.array(enrolled_encoding['encoding'])
                
                # Validate enrolled encoding
                if enrolled_face_encoding.shape[0] != 128:
                    logger.warning(f"Skipping invalid enrolled encoding {index}")
                    continue
                
                # Calculate face distance (lower = more similar)
                distance = face_recognition.face_distance(
                    [enrolled_face_encoding], 
                    captured_face_encoding
                )[0]
                
                # Check if it's a match
                is_match = distance <= custom_tolerance
                
                logger.info(f"Encoding {index + 1}: Distance = {distance:.4f}, Match = {is_match}")
                
                if is_match:
                    match_info = {
                        'id': enrolled_encoding.get('id', index),
                        'distance': float(distance),
                        'confidence': float(1 - distance)  # Higher confidence = better match
                    }
                    matches_found.append(match_info)
                    
                    # Track the best match (lowest distance)
                    if distance < best_distance:
                        best_distance = distance
                        best_match = match_info
            
            except Exception as e:
                logger.error(f"Error comparing encoding {index + 1}: {e}")
                continue
        
        if best_match:
            return jsonify({
                'match': True,
                'matchedEncodingId': best_match['id'],
                'confidence': best_match['confidence'],
                'distance': best_match['distance'],
                'all_matches': matches_found if len(matches_found) > 1 else None
            })
        else:
            return jsonify({
                'match': False,
                'message': 'No matching face found'
            })
    
    except Exception as e:
        logger.error(f"Exception during face verification: {str(e)}")
        return jsonify({'error': 'Face verification failed'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'face-recognition-api'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
