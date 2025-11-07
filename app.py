from flask import Flask, render_template, request, jsonify
import os
from predict import get_classifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize classifier
classifier = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home_page():
    """Home page route"""
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    global classifier
    
    if 'photo' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['photo']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Read image file
            image_bytes = file.read()
            
            # Initialize classifier if not already done
            if classifier is None:
                classifier = get_classifier()
            
            # Make prediction
            result = classifier.predict_from_bytes(image_bytes)
            
            if 'error' in result:
                return jsonify({'error': result['error']}), 500
            
            return jsonify({
                'success': True,
                'prediction': {
                    'class': result['class'],
                    'confidence': round(result['confidence'] * 100, 2),
                    'all_probabilities': result['all_probabilities']
                }
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)

    