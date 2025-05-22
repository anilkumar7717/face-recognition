from flask import Flask, request, jsonify
import face_recognition
from flask_cors import CORS
import requests
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)
CORS(app)

@app.errorhandler(Exception)
def handle_error(e):
    print(f"Global Error occurred: {str(e)}")
    import traceback
    print(traceback.format_exc())
    return jsonify({'error': str(e)}), 500

@app.route('/faceDetection', methods=['POST'])
def verify_face():
    try:
        print("\n🔵 Raw Request Data:", request.data)
        print("🟢 Parsed JSON Data:", request.json)

        data = request.json
        captured_url = data.get('capturedImageUrl')
        enrolled_urls = data.get('employeeImageUrls')

        print(f"📷 Captured Image URL: {captured_url}")
        print(f"👥 Enrolled Image URLs: {enrolled_urls}")

        if not captured_url or not enrolled_urls:
            print("❌ Missing capturedImageUrl or employeeImageUrls")
            return jsonify({'error': 'Missing capturedImageUrl or employeeImageUrls'}), 400

        # Step 1: Fetch and convert captured image
        captured_response = requests.get(captured_url, timeout=10)
        print(f"📥 Captured Image Fetch Status Code: {captured_response.status_code}")
        captured_response.raise_for_status()

        img = Image.open(BytesIO(captured_response.content)).convert('RGB')
        img = ImageOps.exif_transpose(img)  # handle image rotation
        captured_img_array = np.array(img)

        print(f"🖼️ Captured image shape: {captured_img_array.shape}")

        # Step 2: Detect face
        face_locations = face_recognition.face_locations(captured_img_array)
        print(f"📍 Face locations in captured image: {face_locations}")

        captured_face_encodings = face_recognition.face_encodings(captured_img_array)

        if not captured_face_encodings:
            print("⚠️ No face detected in the captured image.")
            return jsonify({
                'error': 'No face detected in the captured image.',
                'advice': 'Make sure your face is well-lit, facing forward, and fully visible in the frame.'
            }), 400

        print("✅ Face detected in captured image. Starting comparison with enrolled images...\n")

        # Step 3: Loop through enrolled images and compare
        for index, enrolled_url in enumerate(enrolled_urls):
            try:
                print(f"🔄 Checking Enrolled Image {index + 1}: {enrolled_url}")
                enrolled_response = requests.get(enrolled_url, timeout=10)
                print(f"  - Status Code: {enrolled_response.status_code}")
                enrolled_response.raise_for_status()

                enrolled_img = Image.open(BytesIO(enrolled_response.content)).convert('RGB')
                enrolled_img = ImageOps.exif_transpose(enrolled_img)
                enrolled_img_array = np.array(enrolled_img)

                enrolled_face_encodings = face_recognition.face_encodings(enrolled_img_array)

                if not enrolled_face_encodings:
                    print("  ⚠️ No face detected in this enrolled image. Skipping.")
                    continue

                match = face_recognition.compare_faces(captured_face_encodings, enrolled_face_encodings[0], tolerance=0.6)
                print(f"  🔍 Match result: {match[0]}")

                if match[0]:
                    print(f"✅ Match found with enrolled image: {enrolled_url}")
                    return jsonify({'match': True, 'matchedImageUrl': enrolled_url})

            except requests.exceptions.RequestException as e:
                print(f"❌ Error fetching enrolled image from {enrolled_url}: {e}")

        print("❌ No matching face found.")
        return jsonify({'match': False})

    except Exception as e:
        print(f"🔥 Exception occurred during face verification: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# ✅ ONLY THIS ONE run block
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
