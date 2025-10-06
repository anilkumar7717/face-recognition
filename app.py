from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
import numpy as np
import logging
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model as get_insight_model
import cv2

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize face analysis with different model configurations
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Initialize alternative face detector for extreme low-light
face_app_sensitive = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
face_app_sensitive.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.3)
# Initialize RetinaFace detector (fallback)
retinaface_detector = None
try:
    retinaface_detector = get_insight_model('retinaface_r50_v1')
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
    
    # 3. Enhanced noise detection using multiple methods
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Additional noise metric using standard deviation of Gaussian difference
    gaussian1 = cv2.GaussianBlur(gray, (3, 3), 0)
    gaussian2 = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_estimate = np.std(gaussian1.astype(np.float32) - gaussian2.astype(np.float32))
    
    # 4. Enhanced blur detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    sharpness = np.mean(gradient_magnitude)
    
    # Alternative sharpness metric using variance of Laplacian
    laplacian_sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 5. Contrast analysis
    contrast = brightness_std
    
    # 6. Local brightness variation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    local_mean = cv2.morphologyEx(gray.astype(np.float32), cv2.MORPH_CLOSE, kernel)
    local_variation = np.std(local_mean)
    
    # 7. Dynamic range analysis
    dynamic_range = np.max(gray) - np.min(gray)
    
    # More aggressive thresholds for low-light detection
    is_extremely_low_light = brightness < 25 or (dark_pixels_ratio > 0.8)
    is_very_low_light = brightness < 45 or (dark_pixels_ratio > 0.7 and bright_pixels_ratio < 0.05)
    is_low_light = brightness < 70 or (dark_pixels_ratio > 0.6 and bright_pixels_ratio < 0.1)
    is_uneven_lighting = local_variation > 20
    is_low_contrast = contrast < 30 or dynamic_range < 80
    is_noisy = laplacian_var < 150 or noise_estimate > 2.0
    is_blurry = sharpness < 25 or laplacian_sharpness < 200
    
    logger.info(f"Quality Analysis - Brightness: {brightness:.1f}, Contrast: {contrast:.1f}, "
                f"Sharpness: {sharpness:.1f}, Dark pixels: {dark_pixels_ratio:.2f}, "
                f"Dynamic range: {dynamic_range:.1f}, Noise estimate: {noise_estimate:.2f}")
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'sharpness': sharpness,
        'dynamic_range': dynamic_range,
        'noise_estimate': noise_estimate,
        'laplacian_sharpness': laplacian_sharpness,
        'is_extremely_low_light': is_extremely_low_light,
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
        'needs_enhancement': is_extremely_low_light or is_very_low_light or is_low_light or is_uneven_lighting or is_low_contrast
    }

def guided_filter_enhancement(img_array, radius=8, eps=0.01):
    """
    Guided filter for edge-preserving smoothing while enhancing details
    """
    try:
        img_float = img_array.astype(np.float32) / 255.0
        gray_float = cv2.cvtColor(img_float, cv2.COLOR_RGB2GRAY)
        
        # Simple guided filter implementation
        mean_I = cv2.boxFilter(gray_float, cv2.CV_32F, (radius, radius))
        mean_p = cv2.boxFilter(img_float, cv2.CV_32F, (radius, radius))
        
        corr_I = cv2.boxFilter(gray_float * gray_float, cv2.CV_32F, (radius, radius))
        var_I = corr_I - mean_I * mean_I
        
        corr_Ip = cv2.boxFilter(np.stack([gray_float] * 3, axis=-1) * img_float, cv2.CV_32F, (radius, radius))
        cov_Ip = corr_Ip - np.stack([mean_I] * 3, axis=-1) * mean_p
        
        a = cov_Ip / (var_I[..., np.newaxis] + eps)
        b = mean_p - a * np.stack([mean_I] * 3, axis=-1)
        
        mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
        
        result = mean_a * np.stack([gray_float] * 3, axis=-1) + mean_b
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in guided filter: {e}")
        return img_array

def adaptive_histogram_stretch(img_array, low_percentile=2, high_percentile=98):
    """
    Adaptive histogram stretching for better contrast
    """
    try:
        img_float = img_array.astype(np.float32)
        
        # Calculate percentiles for each channel
        stretched = np.zeros_like(img_float)
        
        for i in range(3):  # RGB channels
            channel = img_float[:, :, i]
            low_val = np.percentile(channel, low_percentile)
            high_val = np.percentile(channel, high_percentile)
            
            if high_val > low_val:
                stretched[:, :, i] = 255 * (channel - low_val) / (high_val - low_val)
            else:
                stretched[:, :, i] = channel
                
        stretched = np.clip(stretched, 0, 255).astype(np.uint8)
        return stretched
        
    except Exception as e:
        logger.error(f"Error in histogram stretching: {e}")
        return img_array

def extreme_low_light_enhancement(img_array):
    """
    Extreme enhancement for very dark images with multiple techniques
    """
    try:
        logger.info("Applying extreme low-light enhancement...")
        
        # Stage 1: Adaptive histogram stretching
        enhanced = adaptive_histogram_stretch(img_array, low_percentile=1, high_percentile=99)
        
        # Stage 2: Multi-scale Retinex
        enhanced = multi_scale_retinex(enhanced, scales=[15, 80, 250])
        
        # Stage 3: Aggressive CLAHE in multiple color spaces
        # LAB space CLAHE
        img_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
        l_enhanced = clahe.apply(l)
        
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        enhanced = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        
        # Stage 4: Guided filter for noise reduction while preserving edges
        enhanced = guided_filter_enhancement(enhanced, radius=6, eps=0.01)
        
        # Stage 5: Adaptive gamma correction based on current brightness
        current_brightness = np.mean(cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY))
        if current_brightness < 40:
            gamma = 0.4
        elif current_brightness < 60:
            gamma = 0.6
        else:
            gamma = 0.8
            
        enhanced = np.power(enhanced / 255.0, gamma)
        enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
        
        # Stage 6: Final sharpening
        enhanced = adaptive_unsharp_mask(enhanced, strength=0.5)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Error in extreme low-light enhancement: {e}")
        return img_array

def multi_scale_retinex(img_array, scales=[15, 80, 250]):
    """
    Multi-Scale Retinex for better illumination normalization
    """
    try:
        img_float = img_array.astype(np.float32) + 1.0
        retinex = np.zeros_like(img_float)
        
        for scale in scales:
            # Gaussian blur
            blurred = cv2.GaussianBlur(img_float, (0, 0), scale)
            
            # Single Scale Retinex
            ssr = np.log(img_float) - np.log(blurred)
            retinex += ssr
            
        retinex = retinex / len(scales)
        
        # Normalize
        for i in range(3):
            channel = retinex[:, :, i]
            channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel))
            retinex[:, :, i] = channel
            
        retinex = (retinex * 255).astype(np.uint8)
        return retinex
        
    except Exception as e:
        logger.error(f"Error in multi-scale Retinex: {e}")
        return img_array

def adaptive_unsharp_mask(img_array, strength=0.5, radius=2.0, threshold=0):
    """
    Adaptive unsharp masking for sharpening
    """
    try:
        # Create Gaussian blur
        blurred = cv2.GaussianBlur(img_array, (0, 0), radius)
        
        # Create unsharp mask
        mask = img_array.astype(np.float32) - blurred.astype(np.float32)
        
        # Adaptive strength based on local variance
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        local_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if local_var < 100:
            adaptive_strength = strength * 1.5
        else:
            adaptive_strength = strength * 0.8
        
        # Apply threshold
        if threshold > 0:
            mask = np.where(np.abs(mask) < threshold, 0, mask)
        
        # Apply mask
        sharpened = img_array.astype(np.float32) + adaptive_strength * mask
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
        
    except Exception as e:
        logger.error(f"Error in adaptive unsharp mask: {e}")
        return img_array

def illumination_normalization(img_array):
    """
    Homomorphic filtering for illumination normalization
    """
    try:
        # Convert to float and add small constant to avoid log(0)
        img_float = img_array.astype(np.float32) + 1.0
        
        # Take logarithm
        log_img = np.log(img_float)
        
        # Apply high-pass filter in frequency domain
        # For simplicity, use Gaussian high-pass filter
        rows, cols = log_img.shape[:2]
        crow, ccol = rows // 2, cols // 2
        
        # Process each channel
        result = np.zeros_like(log_img)
        for i in range(3):
            # FFT
            f_transform = np.fft.fft2(log_img[:, :, i])
            f_shift = np.fft.fftshift(f_transform)
            
            # Create high-pass filter
            mask = np.ones((rows, cols), np.float32)
            r = 30  # radius for high-pass
            y, x = np.ogrid[:rows, :cols]
            mask_area = (x - ccol)**2 + (y - crow)**2 <= r**2
            mask[mask_area] = 0.3  # Don't completely remove low frequencies
            
            # Apply filter
            f_shift_filtered = f_shift * mask
            
            # Inverse FFT
            f_ishift = np.fft.ifftshift(f_shift_filtered)
            result[:, :, i] = np.real(np.fft.ifft2(f_ishift))
        
        # Exponential and normalize
        result = np.exp(result) - 1.0
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in illumination normalization: {e}")
        return img_array

def smart_image_enhancement(img_array):
    """
    Smart enhancement with specialized low-light processing
    """
    try:
        quality_issues = detect_image_quality_issues(img_array)
        enhanced_img = img_array.copy()
        
        logger.info(f"Enhancement strategy - Extremely low light: {quality_issues['is_extremely_low_light']}, "
                   f"Very low light: {quality_issues['is_very_low_light']}, "
                   f"Low light: {quality_issues['is_low_light']}")
        
        # For extremely low light conditions
        if quality_issues['is_extremely_low_light']:
            logger.info("Applying extreme low-light enhancement...")
            enhanced_img = extreme_low_light_enhancement(enhanced_img)
            
            # Additional illumination normalization
            logger.info("Applying illumination normalization...")
            enhanced_img = illumination_normalization(enhanced_img)
            
        # For very low light conditions
        elif quality_issues['is_very_low_light']:
            logger.info("Applying advanced low-light enhancement...")
            enhanced_img = advanced_low_light_enhancement(enhanced_img)
            
        # For regular low light
        elif quality_issues['is_low_light']:
            logger.info("Applying moderate low-light enhancement...")
            enhanced_img = moderate_low_light_enhancement(enhanced_img)
        
        # Handle other quality issues
        if quality_issues['is_noisy']:
            logger.info("Applying noise reduction...")
            enhanced_img = cv2.bilateralFilter(enhanced_img, 7, 50, 50)
        
        if quality_issues['is_blurry']:
            logger.info("Applying sharpening...")
            enhanced_img = adaptive_unsharp_mask(enhanced_img, strength=0.4)
        
        return enhanced_img
        
    except Exception as e:
        logger.error(f"Error in smart enhancement: {e}")
        return img_array

def advanced_low_light_enhancement(img_array):
    """
    Advanced multi-stage low-light enhancement
    """
    try:
        logger.info("Applying advanced low-light enhancement...")
        
        # Stage 1: Multi-scale Retinex
        enhanced = multi_scale_retinex(img_array, scales=[20, 100, 200])
        
        # Stage 2: Adaptive histogram equalization
        img_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        brightness = np.mean(l)
        if brightness < 40:
            clip_limit = 6.0
            tile_size = (4, 4)
        elif brightness < 60:
            clip_limit = 4.0
            tile_size = (6, 6)
        else:
            clip_limit = 3.0
            tile_size = (8, 8)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l_enhanced = clahe.apply(l)
        
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        enhanced = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        
        # Stage 3: Guided filtering
        enhanced = guided_filter_enhancement(enhanced, radius=8, eps=0.02)
        
        # Stage 4: Adaptive gamma correction
        current_brightness = np.mean(cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY))
        if current_brightness < 50:
            gamma = 0.6
        elif current_brightness < 70:
            gamma = 0.75
        else:
            gamma = 0.9
            
        enhanced = np.power(enhanced / 255.0, gamma)
        enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Error in advanced low-light enhancement: {e}")
        return img_array

def moderate_low_light_enhancement(img_array):
    """
    Moderate enhancement for mild low-light conditions
    """
    try:
        # Histogram stretching
        enhanced = adaptive_histogram_stretch(img_array, low_percentile=5, high_percentile=95)
        
        # Mild CLAHE
        img_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        enhanced = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        
        # Mild gamma correction
        enhanced = np.power(enhanced / 255.0, 0.85)
        enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Error in moderate low-light enhancement: {e}")
        return img_array

def normalize_brightness(img_array, target_mean: float = 120.0, min_factor: float = 0.5, max_factor: float = 2.5):
    """
    Normalize global brightness with more aggressive scaling for low-light
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
        
        logger.info(f"Brightness normalization: {current_mean:.1f} -> {np.mean(cv2.cvtColor(img_norm, cv2.COLOR_RGB2GRAY)):.1f} (scale: {scale:.2f})")
        
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
    logger.info(f"Original image shape: {img_array.shape}, brightness: {np.mean(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)):.1f}")

    # Compute quality on original
    quality_issues = detect_image_quality_issues(img_array)
    
    # Apply smart enhancement
    logger.info("Analyzing image for enhancement...")
    enhanced_img = smart_image_enhancement(img_array)

    # More aggressive brightness normalization for low-light
    if quality_issues['is_extremely_low_light']:
        normalized_img = normalize_brightness(enhanced_img, target_mean=140.0, max_factor=3.0)
    elif quality_issues['is_very_low_light']:
        normalized_img = normalize_brightness(enhanced_img, target_mean=130.0, max_factor=2.5)
    else:
        normalized_img = normalize_brightness(enhanced_img, target_mean=120.0)
    
    logger.info(f"Final processed brightness: {np.mean(cv2.cvtColor(normalized_img, cv2.COLOR_RGB2GRAY)):.1f}")
    
    return normalized_img, quality_issues

def get_best_face_embedding(img_array, retry_with_enhancement=True):
    """
    Enhanced face detection with progressive low-light strategies
    """
    try:
        # First try with the already enhanced image using standard detector
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        faces = face_app.get(img_bgr)

        if faces:
            logger.info(f"Face detected with standard detector! Found {len(faces)} face(s)")
            return select_best_face(faces)
        
        # Try with sensitive detector (lower threshold, smaller input size)
        logger.info("Trying sensitive face detector...")
        faces = face_app_sensitive.get(img_bgr)
        
        if faces:
            logger.info(f"Face detected with sensitive detector! Found {len(faces)} face(s)")
            return select_best_face(faces)
        
        if not retry_with_enhancement:
            return None
            
        logger.info("No faces detected, trying progressive enhancement strategies...")
        
        # Strategy 1: Extreme enhancement + standard detector
        strategy1_img = extreme_low_light_enhancement(img_array)
        strategy1_bgr = cv2.cvtColor(strategy1_img, cv2.COLOR_RGB2BGR)
        faces = face_app.get(strategy1_bgr)
        
        if faces:
            logger.info(f"Face detected with extreme enhancement + standard detector! Found {len(faces)} face(s)")
            return select_best_face(faces)
        
        # Strategy 2: Extreme enhancement + sensitive detector
        faces = face_app_sensitive.get(strategy1_bgr)
        
        if faces:
            logger.info(f"Face detected with extreme enhancement + sensitive detector! Found {len(faces)} face(s)")
            return select_best_face(faces)
        
        # Strategy 3: Multiple preprocessing variations
        variations = [
            ("high_contrast", high_contrast_enhancement(img_array)),
            ("edge_preserving", edge_preserving_enhancement(img_array)),
            ("illumination_normalized", illumination_normalization(img_array)),
            ("histogram_stretched", adaptive_histogram_stretch(img_array, 1, 99))
        ]
        
        for name, variant_img in variations:
            variant_bgr = cv2.cvtColor(variant_img, cv2.COLOR_RGB2BGR)
            
            # Try both detectors
            for detector_name, detector in [("standard", face_app), ("sensitive", face_app_sensitive)]:
                faces = detector.get(variant_bgr)
                if faces:
                    logger.info(f"Face detected with {name} + {detector_name} detector! Found {len(faces)} face(s)")
                    return select_best_face(faces)
        
        # Strategy 4: Multi-scale detection
        logger.info("Trying multi-scale detection...")
        scales = [1.0, 1.2, 0.8, 1.5, 0.6]
        for scale in scales:
            if scale != 1.0:
                h, w = img_array.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_img = cv2.resize(img_array, (new_w, new_h))
                scaled_bgr = cv2.cvtColor(scaled_img, cv2.COLOR_RGB2BGR)
                
                for detector_name, detector in [("standard", face_app), ("sensitive", face_app_sensitive)]:
                    faces = detector.get(scaled_bgr)
                    if faces:
                        logger.info(f"Face detected with scale {scale} + {detector_name} detector! Found {len(faces)} face(s)")
                        return select_best_face(faces)
        
        # Strategy 5: RetinaFace fallback with enhanced images
        if retinaface_detector is not None:
            logger.info("Trying RetinaFace fallback detection...")
            
            # Try RetinaFace on various enhanced versions
            test_images = [
                ("original", img_array),
                ("extreme_enhanced", extreme_low_light_enhancement(img_array)),
                ("high_contrast", high_contrast_enhancement(img_array))
            ]
            
            for name, test_img in test_images:
                try:
                    test_bgr = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
                    bboxes, landmarks = retinaface_detector.detect(test_bgr, threshold=0.2, scale=1.0)

                    if bboxes is not None and len(bboxes) > 0:
                        logger.info(f"RetinaFace detected faces on {name} image")
                        # Pick the largest bbox
                        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                        best_idx = int(np.argmax(areas))
                        x1, y1, x2, y2 = bboxes[best_idx][:4].astype(int)

                        # Expand ROI
                        h, w = test_img.shape[:2]
                        pad_x = int(0.2 * (x2 - x1))
                        pad_y = int(0.25 * (y2 - y1))
                        xx1 = max(0, x1 - pad_x)
                        yy1 = max(0, y1 - pad_y)
                        xx2 = min(w, x2 + pad_x)
                        yy2 = min(h, y2 + pad_y)

                        face_roi_rgb = test_img[yy1:yy2, xx1:xx2]
                        if face_roi_rgb.size > 0:
                            roi_bgr = cv2.cvtColor(face_roi_rgb, cv2.COLOR_RGB2BGR)
                            
                            # Try both detectors on ROI
                            for detector_name, detector in [("standard", face_app), ("sensitive", face_app_sensitive)]:
                                faces_roi = detector.get(roi_bgr)
                                if faces_roi:
                                    logger.info(f"Face detected via RetinaFace ROI + {detector_name} detector! Found {len(faces_roi)} face(s)")
                                    return select_best_face(faces_roi)
                except Exception as e:
                    logger.warning(f"RetinaFace error on {name}: {e}")

        logger.warning("No faces detected even after all enhancement strategies")
        return None

    except Exception as e:
        logger.error(f"Error in face embedding: {e}")
        return None

def high_contrast_enhancement(img_array):
    """
    High contrast enhancement for better feature detection
    """
    try:
        # Multiple contrast enhancement techniques
        # 1. Histogram equalization
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        equalized = cv2.equalizeHist(gray)
        
        # 2. Convert back to RGB and blend
        enhanced_rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
        enhanced_rgb = cv2.addWeighted(img_array, 0.4, enhanced_rgb, 0.6, 0)
        
        # 3. Additional contrast stretching
        enhanced_rgb = adaptive_histogram_stretch(enhanced_rgb, 3, 97)
        
        return enhanced_rgb
        
    except Exception as e:
        logger.error(f"Error in high contrast enhancement: {e}")
        return img_array

def edge_preserving_enhancement(img_array):
    """
    Edge-preserving enhancement to maintain facial structure
    """
    try:
        # Edge-preserving filter with stronger parameters
        enhanced = cv2.edgePreservingFilter(img_array, flags=2, sigma_s=50, sigma_r=0.3)
        
        # Additional brightness boost with contrast preservation
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.4, beta=30)
        
        # Apply guided filter for further enhancement
        enhanced = guided_filter_enhancement(enhanced, radius=4, eps=0.01)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Error in edge-preserving enhancement: {e}")
        return img_array

def select_best_face(faces):
    """
    Select the best face from detected faces with improved scoring
    """
    faces_with_scores = []
    for face in faces:
        bbox_area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
        
        # Base score from area
        score = bbox_area
        
        # Weight by detection score if available
        if hasattr(face, 'det_score'):
            score *= face.det_score
        
        # Prefer faces closer to center of image (often better quality)
        bbox_center_x = (face.bbox[0] + face.bbox[2]) / 2
        bbox_center_y = (face.bbox[1] + face.bbox[3]) / 2
        
        # Assume image dimensions (this could be improved by passing image shape)
        # For now, we'll use a simple heuristic
        center_bonus = 1.0 - (abs(bbox_center_x - 400) + abs(bbox_center_y - 400)) / 800
        center_bonus = max(0.5, center_bonus)  # Don't penalize too much
        
        score *= center_bonus
        
        faces_with_scores.append((face, score))
    
    # Get the best face
    best_face = max(faces_with_scores, key=lambda x: x[1])[0]
    
    logger.info(f"Selected best face with bbox area: {(best_face.bbox[2] - best_face.bbox[0]) * (best_face.bbox[3] - best_face.bbox[1]):.2f}, "
               f"det_score: {getattr(best_face, 'det_score', 'N/A')}")
    
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

        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
        file_ext = image_file.filename.rsplit('.', 1)[1].lower() if '.' in image_file.filename else ''
        if file_ext not in allowed_extensions:
            resp = {'error': 'Invalid file type'}
            logger.info(f"Returning: {resp}")
            return jsonify(resp), 400

        # Reset file pointer
        image_file.seek(0)
        img_array, quality_issues = preprocess_image(image_file)
        face_embedding = get_best_face_embedding(img_array)

        if face_embedding is None:
            # Provide more specific error messages based on detected issues
            if quality_issues.get('is_extremely_low_light'):
                error_msg = 'No face detected. The image is extremely dark. Please try to improve lighting conditions or use a flash.'
            elif quality_issues.get('is_very_low_light'):
                error_msg = 'No face detected. The image is very dark. Please try to improve lighting or move closer to a light source.'
            elif quality_issues.get('is_low_light'):
                error_msg = 'No face detected. The image appears to be in low light. Please try to improve lighting conditions.'
            elif quality_issues.get('is_blurry'):
                error_msg = 'No face detected. The image appears blurry. Please ensure the camera is steady and in focus.'
            else:
                error_msg = 'No face detected. Please ensure the face is clearly visible and well-lit.'
                
            resp = {'error': error_msg}
            logger.info(f"Returning: {resp}")
            return jsonify(resp), 400

        # Determine encoding quality based on image conditions
        if quality_issues.get('is_extremely_low_light') or quality_issues.get('brightness', 0) < 30:
            encoding_quality = 'poor_lighting'
        elif quality_issues.get('is_very_low_light') or quality_issues.get('brightness', 0) < 50:
            encoding_quality = 'low_lighting'
        elif quality_issues.get('is_low_light') or quality_issues.get('brightness', 0) < 70:
            encoding_quality = 'moderate_lighting'
        else:
            encoding_quality = 'good'

        resp = {
            'success': True,
            'face_encoding': face_embedding,
            'encoding_quality': encoding_quality,
            'lighting': {
                'is_low_light': bool(quality_issues.get('is_low_light', False)),
                'is_very_low_light': bool(quality_issues.get('is_very_low_light', False)),
                'is_extremely_low_light': bool(quality_issues.get('is_extremely_low_light', False)),
                'brightness': float(quality_issues.get('brightness', 0.0)),
                'contrast': float(quality_issues.get('contrast', 0.0)),
                'dynamic_range': float(quality_issues.get('dynamic_range', 0.0))
            },
            'image_quality': {
                'sharpness': float(quality_issues.get('sharpness', 0.0)),
                'is_blurry': bool(quality_issues.get('is_blurry', False)),
                'is_noisy': bool(quality_issues.get('is_noisy', False)),
                'needs_enhancement': bool(quality_issues.get('needs_enhancement', False))
            }
        }
        logger.info(f"Successfully generated face encoding with quality: {encoding_quality}")
        return jsonify(resp)

    except Exception as e:
        logger.error(f"Error converting image to face encoding: {str(e)}")
        return jsonify({'error': 'Failed to process image. Please try again with a clearer image.'}), 500

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
        image_quality = data.get('image_quality', {}) or {}

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

                # Calculate cosine similarity
                dot_product = np.dot(captured_encoding, enrolled_encoding)
                norm_product = np.linalg.norm(captured_encoding) * np.linalg.norm(enrolled_encoding)
                similarity = dot_product / norm_product
                distance = 1 - similarity

                logger.info(f"Encoding {index + 1} (ID={enrolled.get('id', index)}): Distance = {distance:.4f}, Similarity = {similarity:.4f}")

                if distance < best_distance:
                    best_distance = distance
                    best_match = {
                        'id': enrolled.get('id', index),
                        'distance': float(distance),
                        'similarity': float(similarity)
                    }

            except Exception as e:
                logger.error(f"Error comparing encoding at index {index}: {e}")
                continue

        if best_match:
            distance_value = float(best_distance)
            confidence = 1.0 - distance_value
            match_percentage = round(confidence * 100, 2)

            # Enhanced threshold policy based on image quality
            is_extremely_low_light = bool(lighting_info.get('is_extremely_low_light', False))
            is_very_low_light = bool(lighting_info.get('is_very_low_light', False))
            is_low_light = bool(lighting_info.get('is_low_light', False))
            is_poor_quality = bool(image_quality.get('is_blurry', False)) or bool(image_quality.get('is_noisy', False))
            
            brightness = lighting_info.get('brightness', 100)
            
            # Adaptive threshold based on conditions
            if is_extremely_low_light or brightness < 30:
                confidence_threshold = 0.68  # Very lenient for extreme low light
                threshold_reason = "extreme_low_light"
            elif is_very_low_light or brightness < 50:
                confidence_threshold = 0.70  # Lenient for very low light
                threshold_reason = "very_low_light"
            elif is_low_light or brightness < 70 or is_poor_quality:
                confidence_threshold = 0.70  # Moderate for low light/poor quality
                threshold_reason = "low_light_or_poor_quality"
            else:
                # For good lighting, check if score suggests hidden low-light conditions
                if 0.75 <= confidence < 0.72:
                    confidence_threshold = 0.70
                    threshold_reason = "assumed_low_light"
                else:
                    confidence_threshold = 0.72  # Standard threshold for good conditions
                    threshold_reason = "normal"
            
            is_match = confidence >= confidence_threshold
            
            if is_match: 
                resp = {
                    'match': True,
                    'matchedEncodingId': best_match['id'],
                    'distance': distance_value,
                    'confidence': round(confidence, 3),
                    'match_percentage': match_percentage,
                    'used_threshold': round(confidence_threshold * 100, 2),
                    'threshold_reason': threshold_reason,
                    'lighting_conditions': {
                        'extremely_low_light': is_extremely_low_light,
                        'very_low_light': is_very_low_light,
                        'low_light': is_low_light,
                        'brightness': brightness
                    },
                    'image_quality': image_quality
                } 
                logger.info(f"MATCH FOUND: {resp}")
                return jsonify(resp)
            else:
                resp = {
                    'match': False,
                    'message': f'Face found, but confidence is below {round(confidence_threshold*100, 2)}%. '
                             f'Threshold adjusted for {threshold_reason.replace("_", " ")} conditions.',
                    'min_distance': distance_value,
                    'confidence': round(confidence, 3),
                    'match_percentage': match_percentage,
                    'used_threshold': round(confidence_threshold * 100, 2),
                    'threshold_reason': threshold_reason,
                    'lighting_conditions': {
                        'extremely_low_light': is_extremely_low_light,
                        'very_low_light': is_very_low_light,
                        'low_light': is_low_light,
                        'brightness': brightness
                    },
                    'image_quality': image_quality
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)