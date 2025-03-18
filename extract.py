from flask import Flask, request, jsonify
from flask_cors import CORS
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import os
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the OCR model when the application starts
model = ocr_predictor(pretrained=True)

@app.route('/extract-text', methods=['POST'])
def extract_text():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    
    # Use temporary file instead of fixed filename
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        image_path = temp_file.name
        image_file.save(image_path)

    try:
        # Process the image with Doctr
        doc = DocumentFile.from_images(image_path)
        result = model(doc)

        # Extract text
        extracted_text = " ".join([word.value for page in result.pages 
                                 for block in page.blocks 
                                 for line in block.lines 
                                 for word in line.words])

        print(extracted_text)

        return jsonify({"extracted_text": extracted_text})
    
    finally:
        # Cleanup
        if os.path.exists(image_path):
            os.remove(image_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)