

import streamlit
import tempfile
from transformers import pipeline
import fitz
from dotenv import load_dotenv
import os
from openai import OpenAI
import base64
# Load environment variables
load_dotenv() 

model_name = "openai/clip-vit-large-patch14-336"
classifier = pipeline("zero-shot-image-classification", model=model_name)
labels = ["Transaction receipt", "other"]
CONFIDENCE_THRESHOLD = 0.7


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_transaction_data(image_path):
    base64_image = encode_image_to_base64(image_path)
    prompt = """Extract the following transaction details:
    - Date and Time
    - Amount
    - Merchant/Recipient
    - Transaction Type
    - Reference Number (if any)
    Return in a clear, structured format."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }],
        max_tokens=300
    )
    return response.choices[0].message.content

def process_image(file, output_folder="converted_images"):
    os.makedirs(output_folder, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
        tmp_file.write(file.getvalue())
        file_path = tmp_file.name

    file_extension = file.name.lower().split(".")[-1]
    if file_extension in {"jpg", "jpeg", "png"}:
        return file_path
    elif file_extension == "pdf":
        doc = fitz.open(file_path)
        page = doc.load_page(0)
        pix = page.get_pixmap()
        output_path = os.path.join(output_folder, f"converted_page_1.png")
        pix.save(output_path)
        doc.close()
        return output_path
    return None

streamlit.set_page_config(page_title="Receipt Analyzer")
streamlit.write("# Receipt Analyzer")

if "messages" not in streamlit.session_state:
    streamlit.session_state.messages = []

uploaded_file = streamlit.file_uploader("Upload receipt", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file:
    streamlit.image(uploaded_file)
    processed_file = process_image(uploaded_file)
    
    with streamlit.spinner("Validating receipt..."):
        scores = classifier(processed_file, candidate_labels=labels)
        is_receipt = scores[0]['label'] == "Transaction receipt" and scores[0]['score'] > CONFIDENCE_THRESHOLD

    if is_receipt:
        streamlit.success("✅ Valid transaction receipt")
        with streamlit.spinner("Extracting transaction details..."):
            transaction_data = extract_transaction_data(processed_file)
            streamlit.write("## Transaction Details")
            streamlit.write(transaction_data)
        
        streamlit.write("## Ask Questions")
        if prompt := streamlit.chat_input("Ask about specific details"):
            streamlit.chat_message("user").write(prompt)
            streamlit.session_state.messages.append({"role": "user", "content": prompt})
            
            with streamlit.spinner("Analyzing..."):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encode_image_to_base64(processed_file)}"
                                }
                            }
                        ]
                    }],
                    max_tokens=300
                ).choices[0].message.content
                
            streamlit.chat_message("assistant").write(response)
            streamlit.session_state.messages.append({"role": "assistant", "content": response})
    else:
        streamlit.error("❌ This document does not appear to be a valid transaction receipt")