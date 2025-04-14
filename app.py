from flask import Flask, request, jsonify
import pdfplumber
import os
from openai import AzureOpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
# from tika import parser
import re
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("nbroad/ESG-BERT")

model = AutoModelForSequenceClassification.from_pretrained("nbroad/ESG-BERT")

# Create the pipeline for text classification
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

endpoint = "https://esg-finetune.openai.azure.com/"
model_name = "gpt-4o-mini"
deployment = "gpt-4o-mini"

subscription_key = "4XlXF0Y7Xxp2mM6fnI6LmdUwd6GsZLMW5DsDKHb34stBas7cSflkJQQJ99BDACHrzpqXJ3w3AAABACOGveCN"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

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
        print("File found")
        # Use pdfplumber to open and extract text from the PDF file directly from the file stream
        with pdfplumber.open(file) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        full_text = re.sub(r'\n', ' ', full_text)
        full_text = re.sub(r'\s+', ' ', full_text)
        chunks = []
        print("Processing PDF")
        while len(full_text) > 512:
            # Find the last full stop within the first 512 characters
            split_index = full_text[:512].rfind('.')
            if split_index == -1:
                # If no full stop is found, split at 512 characters
                split_index = 512
            chunks.append(full_text[:split_index + 1].strip())
            full_text = full_text[split_index + 1:].strip()
        # Append any remaining text
        if full_text:
            chunks.append(full_text)
        print("Done processing")
        print(chunks)

        result = classifier(chunks)
        df = pd.DataFrame(result)
        output = df.groupby(['label']).mean().sort_values('score', ascending = False)

        response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an ESG risk analyst.",
            },
            {
                "role": "user",
                "content": f"Given the data in {output}, predict the Total risk, E/S/G Risk for this company. Explain the factors/reasons for the predicted E/S/G scores. And for each E/S/G factor provide 1-2 suggestions to improve. Return the result in json format: total_risk: predicted total risk score from 1-10. 1 is the lowest risk and 10 is the highest risk., e_risk: predicted environmental risk score, e_explanation: reasons for this predicted environmental risk score, e_suggestion: suggestions on how to improve performance in environmental aspect, s_risk: predicted social risk score, s_explanation: reasons for this predicted social risk score, s_suggestion: suggestions on how to improve performance in social aspect, g_risk: predicted governance risk score, g_explanation: reasons for this predicted governance risk score, g_suggestion: suggestions on how to improve performance in governance aspect",
            }
        ],
        max_tokens=4096,
        temperature=0.2,
        top_p=1.0,
        model=deployment
    )
        
        # Return the extracted text as a JSON response
        return jsonify({'result': response.choices[0].message.content}), 200
    except Exception as e:
        # Return the error message if an exception occurs
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
