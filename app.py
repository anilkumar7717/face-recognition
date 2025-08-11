from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import logging
import insightface
from typing import List
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model as get_insight_model
import cv2

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

# Initialize RetinaFace detector (fallback)
retinaface_detector = None
try:
    retinaface_detector = get_insight_model('retinaface_r50_v1')
    # nms default is fine; set up CPU
    retinaface_detector.prepare(ctx_id=0)
    logger.info("RetinaFace fallback detector initialized")
except Exception as e:
    logger.warning(f"Could not initialize RetinaFace fallback detector: {e}")

def detect_image_quality_issues(img_array):
    """
    Enhanced image quality detection with better low-light metrics
    """
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # 1. Enhanced brightness detection
    brightness = np.mean(gray)
    brightness_std = np.std(gray)
    
    # 2. Histogram analysis for lighting conditions
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_normalized = hist / (gray.shape[0] * gray.shape[1])
    
    # Check distribution of pixels across brightness levels
    dark_pixels_ratio = np.sum(hist_normalized[:64])
    mid_pixels_ratio = np.sum(hist_normalized[64:192])
    bright_pixels_ratio = np.sum(hist_normalized[192:])
    
    # 3. Enhanced noise detection
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 4. Enhanced blur detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    sharpness = np.mean(gradient_magnitude)
    
    # 5. Contrast analysis
    contrast = brightness_std
    
    # 6. Local brightness variation (helps detect uneven lighting)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    local_mean = cv2.morphologyEx(gray.astype(np.float32), cv2.MORPH_CLOSE, kernel)
    local_variation = np.std(local_mean)
    
    # Enhanced thresholds for low-light conditions
    is_very_low_light = brightness < 40
    is_low_light = brightness < 80 or (dark_pixels_ratio > 0.6 and bright_pixels_ratio < 0.1)
    is_uneven_lighting = local_variation > 20
    is_low_contrast = contrast < 25
    is_noisy = laplacian_var < 100  # More sensitive for low light
    is_blurry = sharpness < 20  # More sensitive
    
    logger.info(f"Quality Analysis - Brightness: {brightness:.1f}, Contrast: {contrast:.1f}, "
                f"Sharpness: {sharpness:.1f}, Dark pixels: {dark_pixels_ratio:.2f}, "
                f"Local variation: {local_variation:.1f}")
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'sharpness': sharpness,
        'is_very_low_light': is_very_low_light,
        'is_low_light': is_low_light,
        'is_uneven_lighting': is_uneven_lighting,
        'is_low_contrast': is_low_contrast,
        'is_noisy': is_noisy,
        'is_blurry': is_blurry,
        'dark_pixels_ratio': dark_pixels_ratio,
        'mid_pixels_ratio': mid_pixels_ratio,
        'bright_pixels_ratio': bright_pixels_ratio,
        'local_variation': local_variation,
        'needs_enhancement': is_very_low_light or is_low_light or is_uneven_lighting or is_low_contrast or is_noisy or is_blurry
    }

def advanced_low_light_enhancement(img_array):
    """
    Advanced multi-stage low-light enhancement
    """
    try:
        logger.info("Applying advanced low-light enhancement...")
        
        # Convert to different color space for better processing
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Stage 1: Adaptive histogram equalization in LAB space
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Adaptive CLAHE based on current brightness
        brightness = np.mean(l)
        if brightness < 30:
            clip_limit = 4.0
            tile_size = (4, 4)
        elif brightness < 50:
            clip_limit = 3.0
            tile_size = (6, 6)
        elif brightness < 70:
            clip_limit = 2.5
            tile_size = (8, 8)
        else:
            clip_limit = 2.0
            tile_size = (8, 8)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l_enhanced = clahe.apply(l)
        
        # Merge back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        
        # Stage 2: Retinex-inspired enhancement
        enhanced_rgb = retinex_enhancement(enhanced_rgb)
        
        # Stage 3: Adaptive gamma correction
        current_brightness = np.mean(cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY))
        if current_brightness < 60:
            gamma = 0.7
        elif current_brightness < 80:
            gamma = 0.8
        elif current_brightness < 100:
            gamma = 0.9
        else:
            gamma = 1.0
            
        if gamma != 1.0:
            enhanced_rgb = np.power(enhanced_rgb / 255.0, gamma)
            enhanced_rgb = np.clip(enhanced_rgb * 255.0, 0, 255).astype(np.uint8)
        
        # Stage 4: Selective denoising (preserve edges)
        enhanced_rgb = cv2.bilateralFilter(enhanced_rgb, 7, 50, 50)
        
        # Stage 5: Gentle sharpening for low-light recovered details
        enhanced_rgb = adaptive_sharpening(enhanced_rgb, strength=0.3)
        
        return enhanced_rgb
        
    except Exception as e:
        logger.error(f"Error in advanced low-light enhancement: {e}")
        return img_array

def retinex_enhancement(img_array):
    """
    Simplified Single-Scale Retinex for low-light enhancement
    """
    try:
        img_float = img_array.astype(np.float32) + 1.0  # Avoid log(0)
        
        # Apply Gaussian blur to get illumination component
        sigma = 80  # Larger sigma for low-light conditions
        illumination = cv2.GaussianBlur(img_float, (0, 0), sigma)
        
        # Compute Retinex
        retinex = np.log(img_float) - np.log(illumination)
        
        # Normalize and convert back
        retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex))
        retinex = (retinex * 255).astype(np.uint8)
        
        return retinex
        
    except Exception as e:
        logger.error(f"Error in Retinex enhancement: {e}")
        return img_array

def adaptive_sharpening(img_array, strength=0.5):
    """
    Adaptive sharpening that adjusts based on local image properties
    """
    try:
        # Create unsharp mask
        blurred = cv2.GaussianBlur(img_array, (0, 0), 1.0)
        unsharp_mask = img_array - blurred
        
        # Adaptive strength based on local variance
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        local_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if local_variance < 50:  # Low detail areas
            adaptive_strength = strength * 1.5
        else:
            adaptive_strength = strength
        
        sharpened = img_array + adaptive_strength * unsharp_mask
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
        
    except Exception as e:
        logger.error(f"Error in adaptive sharpening: {e}")
        return img_array

def multi_exposure_fusion(img_array):
    """
    Simulate multi-exposure fusion for better low-light performance
    """
    try:
        # Create different "exposures"
        underexposed = np.clip(img_array * 0.7, 0, 255).astype(np.uint8)
        normal = img_array
        overexposed = np.clip(img_array * 1.5, 0, 255).astype(np.uint8)
        
        exposures = [underexposed, normal, overexposed]
        
        # Simple fusion using OpenCV
        merge_mertens = cv2.createMergeMertens()
        exposures_float = [exp.astype(np.float32) / 255.0 for exp in exposures]
        fused = merge_mertens.process(exposures_float)
        
        fused = np.clip(fused * 255, 0, 255).astype(np.uint8)
        return fused
        
    except Exception as e:
        logger.error(f"Error in multi-exposure fusion: {e}")
        return img_array

def smart_image_enhancement(img_array):
    """
    Smart enhancement with specialized low-light processing
    """
    try:
        quality_issues = detect_image_quality_issues(img_array)
        enhanced_img = img_array.copy()
        
        logger.info(f"Enhancement strategy - Very low light: {quality_issues['is_very_low_light']}, "
                   f"Low light: {quality_issues['is_low_light']}, "
                   f"Uneven lighting: {quality_issues['is_uneven_lighting']}")
        
        # For very low light conditions
        if quality_issues['is_very_low_light']:
            logger.info("Applying advanced low-light enhancement...")
            enhanced_img = advanced_low_light_enhancement(enhanced_img)
            
            # Try multi-exposure fusion for extremely dark images
            if quality_issues['brightness'] < 25:
                logger.info("Trying multi-exposure fusion...")
                fusion_result = multi_exposure_fusion(enhanced_img)
                # Use fusion result if it improves brightness significantly
                fusion_brightness = np.mean(cv2.cvtColor(fusion_result, cv2.COLOR_RGB2GRAY))
                current_brightness = np.mean(cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2GRAY))
                if fusion_brightness > current_brightness * 1.2:
                    enhanced_img = fusion_result
                    logger.info("Multi-exposure fusion applied successfully")
        
        # For regular low light
        elif quality_issues['is_low_light']:
            logger.info("Applying moderate low-light enhancement...")
            enhanced_img = advanced_low_light_enhancement(enhanced_img)
        
        # Handle other quality issues
        if quality_issues['is_noisy']:
            logger.info("Applying noise reduction...")
            enhanced_img = cv2.bilateralFilter(enhanced_img, 5, 40, 40)
        
        if quality_issues['is_blurry']:
            logger.info("Applying sharpening...")
            enhanced_img = adaptive_sharpening(enhanced_img, strength=0.4)
        
        return enhanced_img
        
    except Exception as e:
        logger.error(f"Error in smart enhancement: {e}")
        return img_array

def normalize_brightness(img_array, target_mean: float = 110.0, min_factor: float = 0.6, max_factor: float = 1.8):
    """
    Normalize global brightness to a target mean to stabilize embeddings across lighting.
    """
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        current_mean = float(np.mean(gray))
        if current_mean <= 1e-6:
            return img_array

        scale = target_mean / current_mean
        scale = float(np.clip(scale, min_factor, max_factor))

        img_float = img_array.astype(np.float32) * scale
        img_norm = np.clip(img_float, 0, 255).astype(np.uint8)
        return img_norm
    except Exception as e:
        logger.error(f"Error in brightness normalization: {e}")
        return img_array

def preprocess_image(image_file, max_size=(800, 800)):
    """
    Enhanced preprocessing with better low-light handling.
    Returns: (processed_image_array, quality_issues_dict)
    """
    img = Image.open(image_file).convert('RGB')
    img = ImageOps.exif_transpose(img)
    
    # Resize if needed
    if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    img_array = np.array(img)

    # Compute quality on original
    quality_issues = detect_image_quality_issues(img_array)
    
    # Apply smart enhancement
    logger.info("Analyzing image for enhancement...")
    enhanced_img = smart_image_enhancement(img_array)

    # Brightness normalization
    normalized_img = normalize_brightness(enhanced_img)
    
    return normalized_img, quality_issues

def get_best_face_embedding(img_array, retry_with_enhancement=True):
    """
    Enhanced face detection with progressive low-light strategies
    """
    try:
        # First try with the already enhanced image
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        faces = face_app.get(img_bgr)

        if faces:
            logger.info(f"Face detected! Found {len(faces)} face(s)")
            return select_best_face(faces)
        
        if not retry_with_enhancement:
            return None
            
        logger.info("No faces detected, trying aggressive enhancement strategies...")
        
        # Strategy 1: Extreme low-light enhancement
        strategy1_img = extreme_low_light_enhancement(img_array)
        strategy1_bgr = cv2.cvtColor(strategy1_img, cv2.COLOR_RGB2BGR)
        faces = face_app.get(strategy1_bgr)
        
        if faces:
            logger.info(f"Face detected with extreme low-light enhancement! Found {len(faces)} face(s)")
            return select_best_face(faces)
        
        # Strategy 2: High contrast enhancement
        strategy2_img = high_contrast_enhancement(img_array)
        strategy2_bgr = cv2.cvtColor(strategy2_img, cv2.COLOR_RGB2BGR)
        faces = face_app.get(strategy2_bgr)
        
        if faces:
            logger.info(f"Face detected with high contrast enhancement! Found {len(faces)} face(s)")
            return select_best_face(faces)
        
        # Strategy 3: Edge-preserving enhancement
        strategy3_img = edge_preserving_enhancement(img_array)
        strategy3_bgr = cv2.cvtColor(strategy3_img, cv2.COLOR_RGB2BGR)
        faces = face_app.get(strategy3_bgr)
        
        if faces:
            logger.info(f"Face detected with edge-preserving enhancement! Found {len(faces)} face(s)")
            return select_best_face(faces)
        
        # Fallback: RetinaFace detector to get ROI, then embed with InsightFace
        if retinaface_detector is not None:
            try:
                logger.info("No faces with InsightFace. Trying RetinaFace fallback detection...")
                img_for_det = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                bboxes, landmarks = retinaface_detector.detect(img_for_det, threshold=0.25, scale=1.0)

                if bboxes is not None and len(bboxes) > 0:
                    # Pick the largest bbox
                    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                    best_idx = int(np.argmax(areas))
                    x1, y1, x2, y2 = bboxes[best_idx][:4].astype(int)

                    # Expand ROI slightly
                    h, w = img_array.shape[:2]
                    pad_x = int(0.15 * (x2 - x1))
                    pad_y = int(0.20 * (y2 - y1))
                    xx1 = max(0, x1 - pad_x)
                    yy1 = max(0, y1 - pad_y)
                    xx2 = min(w, x2 + pad_x)
                    yy2 = min(h, y2 + pad_y)

                    face_roi_rgb = img_array[yy1:yy2, xx1:xx2]
                    if face_roi_rgb.size > 0:
                        roi_bgr = cv2.cvtColor(face_roi_rgb, cv2.COLOR_RGB2BGR)
                        faces_roi = face_app.get(roi_bgr)
                        if faces_roi:
                            logger.info(f"Face detected via RetinaFace fallback ROI. Found {len(faces_roi)} face(s)")
                            return select_best_face(faces_roi)
                logger.warning("RetinaFace fallback did not detect any faces")
            except Exception as e:
                logger.warning(f"RetinaFace fallback error: {e}")

        logger.warning("No faces detected even after all enhancement strategies and RetinaFace fallback")
        return None

    except Exception as e:
        logger.error(f"Error in face embedding: {e}")
        return None

def extreme_low_light_enhancement(img_array):
    """
    Extreme enhancement for very dark images
    """
    try:
        # More aggressive CLAHE
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
        l_enhanced = clahe.apply(l)
        
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        
        # Aggressive gamma correction
        enhanced_rgb = np.power(enhanced_rgb / 255.0, 0.5)
        enhanced_rgb = np.clip(enhanced_rgb * 255.0, 0, 255).astype(np.uint8)
        
        return enhanced_rgb
        
    except Exception as e:
        logger.error(f"Error in extreme low-light enhancement: {e}")
        return img_array

def high_contrast_enhancement(img_array):
    """
    High contrast enhancement for better feature detection
    """
    try:
        # Convert to grayscale and back for contrast enhancement
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Histogram equalization
        equalized = cv2.equalizeHist(gray)
        
        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
        
        # Blend with original for color preservation
        enhanced_rgb = cv2.addWeighted(img_array, 0.3, enhanced_rgb, 0.7, 0)
        
        return enhanced_rgb
        
    except Exception as e:
        logger.error(f"Error in high contrast enhancement: {e}")
        return img_array

def edge_preserving_enhancement(img_array):
    """
    Edge-preserving enhancement to maintain facial structure
    """
    try:
        # Edge-preserving filter
        enhanced = cv2.edgePreservingFilter(img_array, flags=2, sigma_s=80, sigma_r=0.4)
        
        # Gentle brightness boost
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=20)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Error in edge-preserving enhancement: {e}")
        return img_array

def select_best_face(faces):
    """
    Select the best face from detected faces
    """
    faces_with_scores = []
    for face in faces:
        bbox_area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
        score = bbox_area
        if hasattr(face, 'det_score'):
            score *= face.det_score
        faces_with_scores.append((face, score))
    
    # Get the best face
    best_face = max(faces_with_scores, key=lambda x: x[1])[0]
    
    logger.info(f"Selected best face with bbox area: {(best_face.bbox[2] - best_face.bbox[0]) * (best_face.bbox[3] - best_face.bbox[1]):.2f}")
    
    return best_face.embedding.tolist()

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

        img_array, quality_issues = preprocess_image(image_file)
        face_embedding = get_best_face_embedding(img_array)

        if face_embedding is None:
            resp = {'error': 'No face detected. Please ensure the face is clearly visible. For low-light conditions, try to improve lighting if possible.'}
            logger.info(f"Returning: {resp}")
            return jsonify(resp), 400

        resp = {
            'success': True,
            'face_encoding': face_embedding,
            'encoding_quality': 'good',
            'lighting': {
                'is_low_light': bool(quality_issues.get('is_low_light', False)),
                'is_very_low_light': bool(quality_issues.get('is_very_low_light', False)),
                'brightness': float(quality_issues.get('brightness', 0.0))
            }
        }
        logger.info("Successfully generated face encoding")
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
        lighting_info = data.get('lighting', {}) or {}

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

            # Threshold policy: relax under low-light
            is_low_light_ctx = bool(lighting_info.get('is_low_light')) or bool(lighting_info.get('is_very_low_light'))
            assumed_low_light = False
            if is_low_light_ctx:
                # 75%–79% window requested; use 0.77 as center
                confidence_threshold = 0.77
            else:
                # No low-light flag. If score is in [0.75, 0.82), assume low light and use 0.77
                if 0.75 <= confidence < 0.82:
                    confidence_threshold = 0.77
                    assumed_low_light = True
                else:
                    confidence_threshold = 0.82
            
            if confidence >= confidence_threshold: 
                resp = {
                    'match': True,
                    'matchedEncodingId': best_match['id'],
                    'distance': distance_value,
                    'confidence': round(confidence, 3),
                    'match_percentage': match_percentage,
                    'used_threshold': round(confidence_threshold * 100, 2),
                    'low_light_mode': True if (is_low_light_ctx or assumed_low_light) else False
                } 
                logger.info(f"MATCH FOUND: {resp}")
                return jsonify(resp)
            else:
                resp = {
                    'match': False,
                    'message': f'Face found, but confidence is below {round(confidence_threshold*100, 2)}%. This may be due to lighting conditions.',
                    'min_distance': distance_value,
                    'confidence': round(confidence, 3),
                    'match_percentage': match_percentage,
                    'used_threshold': round(confidence_threshold * 100, 2),
                    'low_light_mode': True if (is_low_light_ctx or assumed_low_light) else False
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
    return jsonify({'status': 'healthy', 'service': 'face-recognition-api-insightface-enhanced-lowlight'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
