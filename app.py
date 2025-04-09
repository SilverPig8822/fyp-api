from flask import Flask, request, jsonify
import pdfplumber

app = Flask(__name__)

@app.route('/extract', methods=['POST'])
def extract_pdf_text():
    # Check if the 'file' part is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    
    file = request.files['file']
    
    # Check if a file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading.'}), 400
    
    try:
        # Use pdfplumber to open and extract text from the PDF file directly from the file stream
        with pdfplumber.open(file) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        
        # Return the extracted text as a JSON response
        return jsonify({'text': full_text}), 200
    except Exception as e:
        # Return the error message if an exception occurs
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
