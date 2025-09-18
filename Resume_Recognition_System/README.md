📄 Resume Recognition System Overview: The Resume Recognition System is a Streamlit-based application that extracts text and images from resumes (PDF) and provides an AI-powered chatbot (via the Groq API) to analyze and answer questions about the parsed data. It is built in Python 3.10 and uses OCR (pytesseract) to extract text from PDF pages converted to images. This system helps recruiters and HR teams convert unstructured resumes into structured, searchable data.

✨ Features -Upload resume files (PDF) -Extract embedded profile images and text using OCR -AI chatbot powered by Groq API to understand the resume -Displays parsed fields: name, email, phone, education, skills, experience -Optionally export results to PDF (reportlab)

⚙ Installation:
1.Install Python 3.10 Download and install Python 3.10 from the official Python website. ⚠ This project must use Python 3.10 (not 3.11+ or 3.13).

2.Clone the repository git clone https://github.com/your-username/resume-recognition-system.git cd resume-recognition-system

3.Install dependencies Install all required libraries globally (no virtual environment required): pip install -r requirements.txt

4.Add your own API key and tweak the code accordingly. eg->(YOUR_API_KEY=your-api-key-here)

Requirements.txt: -streamlit -python-dotenv -pdf2image -pytesseract -Pillow -reportlab -groq

⚡Optional: 
Using a Virtual Environment While not required, you can use a virtual environment (venv) to isolate this project’s dependencies from other Python projects. This helps: Avoid conflicts with other Python packages Make the project more reproducible for other users Keep your system Python environment clean

To create and activate a venv: python3.10 -m venv venv:
Windows
venv\Scripts\activate

macOS/Linux
source venv/bin/activate

Then install dependencies: pip install -r requirements.txt