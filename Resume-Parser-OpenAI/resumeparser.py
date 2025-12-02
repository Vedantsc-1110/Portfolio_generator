import os
import json
from groq import Groq
from PyPDF2 import PdfReader
import docx
from pdf2image import convert_from_path
import pytesseract
# Initialize Groq client
# This tells Python to look for the key in the system settings (Render), not the file
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def ocr_pdf(file_path):
    text = ""
    images = convert_from_path(file_path)

    for img in images:
        text += pytesseract.image_to_string(img)

    return text.strip()

def extract_text_from_pdf(file_path):
    # Try normal extraction first
    text = ""
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted

    text = text.strip()

    # If still no text, try OCR
    if not text:
        print("⚠ No text found! Using OCR...")
        text = ocr_pdf(file_path)

    return text


def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def ats_extractor(file_path):
    # Extract text based on file type
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext in [".docx", ".doc"]:
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload PDF or DOCX.")

    # Prompt for the model
    system_prompt = (
        "You are an ATS resume parser. "
        "Extract all details from the resume and output in valid JSON ONLY. "
        "Do NOT include markdown, explanations, or extra text. "
        "The JSON should strictly follow this format:\n\n"
        "{\n"
        '  "name": "string",\n'
        '  "email": "string",\n'
        '  "phone": "string",\n'
        '  "education": ["list of degrees and institutions"],\n'
        '  "skills": ["list of skills"],\n'
        '  "projects": ["list of key projects with brief details"],\n'
        '  "experience": ["list of work experiences if available"],\n'
        '  "achievements": ["list of awards or achievements"],\n'
        '  "objective": "career objective text"\n'
        "}"
    )

    # Call Groq API
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
    )

    raw_output = response.choices[0].message.content.strip()

    # Parse JSON safely
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        # Try cleaning if model adds extra text
        json_str = raw_output[raw_output.find("{"): raw_output.rfind("}") + 1]
        data = json.loads(json_str)

    return data
