# -------------------------------------
# Import Libraries
# -------------------------------------
import os
import io
import shutil
import subprocess
import requests
import pdfplumber  # Fallback for text extraction
import streamlit as st
from PIL import Image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.colors import green, red, black, HexColor
from dotenv import load_dotenv
from groq import Groq

# -------------------------------------
# Streamlit Page Configuration
# -------------------------------------
st.set_page_config(
    page_title="Resume Recognition System",
    page_icon="📄",
    layout="wide"
)


# Only import pdf2image and pytesseract if dependencies are available
try:
    import pdf2image
    import pytesseract
    USE_PDF2IMAGE = True
except ImportError:
    USE_PDF2IMAGE = False
    st.warning("pdf2image/pytesseract not available. Using pdfplumber as fallback.")

# Attempt to locate binaries from PATH or environment
POPPLER_PATH = os.getenv("POPPLER_PATH")
TESSERACT_CMD = shutil.which("tesseract") or os.getenv("TESSERACT_CMD")

# Windows local fallback paths
if os.name == "nt":
    if not POPPLER_PATH:
        POPPLER_PATH = r"C:\Users\gupta\Desktop\poppler\Release-25.07.0-0\poppler-25.07.0\Library\bin"
    if not TESSERACT_CMD:
        TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Check availability
HAS_POPPLER = POPPLER_PATH and os.path.exists(POPPLER_PATH)
HAS_TESSERACT = TESSERACT_CMD and shutil.which(TESSERACT_CMD)

# Decide whether to use pdf2image + pytesseract
USE_PDF2IMAGE = HAS_POPPLER and HAS_TESSERACT

if not USE_PDF2IMAGE:
    st.warning("⚠️ Poppler and/or Tesseract not found – using text-only fallback (pdfplumber)")

# Configure pytesseract if available
if HAS_TESSERACT:
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    except ImportError:
        st.warning("⚠️ pytesseract not installed, OCR functionality disabled.")

# Import pdf2image only if USE_PDF2IMAGE
if USE_PDF2IMAGE:
    try:
        import pdf2image
    except ImportError:
        st.warning("⚠️ pdf2image not installed, OCR functionality disabled.")
        USE_PDF2IMAGE = False

# -------------------------------------
# Custom CSS for Enhanced UI
# -------------------------------------
st.markdown("""
<style>
/* Dark theme and styling for the app */
.stApp {
    background-color: #0f111a;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.stText, .stMarkdown, .stHeader, .stSubheader, .stTitle, .stButton>button {
    color: #ffffff !important;
}
.stButton>button {
    background-color: #1f77b4;
    color: #ffffff !important;
    border-radius: 8px;
    padding: 0.6em 1.2em;
    font-weight: 600;
    transition: all 0.3s ease-in-out;
    border: none;
    box-shadow: 0 2px 8px rgba(31, 119, 180, 0.3);
}
.stButton>button:hover {
    background-color: #105599;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(31, 119, 180, 0.4);
}
/* Custom styling for analysis results */
.analysis-result {
    background: linear-gradient(135deg, #1c1f2a 0%, #232738 100%);
    color: #ffffff;
    padding: 25px;
    border-left: 6px solid #1f77b4;
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin: 20px 0;
    line-height: 1.6;
    backdrop-filter: blur(10px);
    text-align: left !important;
}
/* Section headers */
.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #1f77b4;
    margin: 25px 0 15px 0;
    border-bottom: 2px solid #1f77b4;
    padding-bottom: 8px;
}
/* Input fields */
textarea, input {
    color: #ffffff !important;
    background-color: #1c1f2a !important;
    border: 1px solid #33364a;
    border-radius: 8px;
}
/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #2a2d3a 0%, #3a3d4a 100%);
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    border: 1px solid #33364a;
    text-align: center;
}
.metric-value {
    font-size: 2.5em;
    font-weight: bold;
    color: #1f77b4;
}
.metric-label {
    font-size: 0.9em;
    color: #888;
    margin-top: 5px;
}
/* Info boxes */
.info-box {
    background: rgba(31, 119, 180, 0.1);
    border: 1px solid rgba(31, 119, 180, 0.3);
    border-radius: 8px;
    padding: 15px;
    margin: 15px 0;
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------
# Initialize Session State
# -------------------------------------
def initialize_session_state():
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Upload & Analyze"
    if "resume_text" not in st.session_state:
        st.session_state.resume_text = ""
    if "job_description" not in st.session_state:
        st.session_state.job_description = ""
    if "preview_images" not in st.session_state:
        st.session_state.preview_images = []
    if "selected_analysis" not in st.session_state:
        st.session_state.selected_analysis = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {}

initialize_session_state()

# -------------------------------------
# Load API Key and Test Connection
# -------------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ Missing Groq API key in .env file")
    st.info("💡 Create a .env file with: GROQ_API_KEY=your_api_key_here")
    st.info("🔗 Get your free API key from: https://console.groq.com/keys")

def call_groq_api(api_key, model, messages, max_tokens=10):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

if groq_api_key and "api_tested" not in st.session_state:
    try:
        response = call_groq_api(
            api_key=groq_api_key,
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        st.session_state.api_tested = True
        st.session_state.api_working = True
        st.success("✅ API connection successful!")
    except Exception as e:
        st.session_state.api_tested = True
        st.session_state.api_working = False
        st.warning(f"⚠ API connection issue: {str(e)}")
        st.info("💡 Please check your API key and try again")

# -------------------------------------
# Helper Functions
# -------------------------------------
def provide_manual_analysis_tips() -> dict:
    return {
        "Quick Overview": """
        Your resume has been uploaded and processed successfully.
        Here are some general guidelines to review:
        - Your contact information is prominent and current.
        - Your experience is listed in reverse chronological order.
        - Your skills section matches the job requirements.
        - Look for quantifiable achievements and results.
        """,
        "Issues Analysis": """
        Common resume issues to check manually:
        - Formatting problems: Inconsistent fonts, sizes, or spacing.
        - Content issues: Typos, vague job descriptions, missing achievements.
        - Structure problems: No clear summary, poor organization.
        """,
        "Enhancement Tips": """
        Focus on these improvement areas:
        - Add quantifiable achievements.
        - Use action verbs to start bullet points.
        - Customize your resume for each job application.
        - Include relevant keywords from the job posting.
        """,
        "Job Matching": """
        To manually assess job matching:
        - Compare your resume against the job posting.
        - Highlight matching skills and experience.
        - Identify gaps: Missing technical skills, lack of industry experience.
        """
    }

def analyze_resume_with_llm(prompt: str, max_retries: int = 3) -> str:
    if not groq_api_key:
        return "⚠ No Groq API key found. Please check your .env file."
    if st.session_state.get('api_tested', False) and not st.session_state.get('api_working', True):
        st.warning("🔄 API connection issues detected. Showing manual analysis guidelines.")
        return "API temporarily unavailable. Please refer to manual analysis guidelines below."

    for attempt in range(max_retries):
        try:
            response = call_groq_api(
                api_key=groq_api_key,
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": """
                    You are a professional resume analyzer and career counselor.
                    Provide detailed, actionable feedback. Use clear formatting with bullet points.
                    For strengths, start bullet points with [STRENGTH].
                    For weaknesses, start bullet points with [WEAKNESS].
                    Be specific and provide concrete examples.
                    """},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            if response and "choices" in response:
                result = response["choices"][0]["message"]["content"]
                if result and len(result.strip()) > 10:
                    return result.strip()
        except Exception as e:
            error_msg = str(e)
            st.session_state._last_groq_error = f"Attempt {attempt+1}: {error_msg}"
            if attempt < max_retries - 1:
                st.info(f"Retrying analysis (attempt {attempt + 2}/{max_retries})...")
                continue
            else:
                st.error(f"❌ Analysis failed: {st.session_state._last_groq_error}")
                st.info("💡 This could be due to:")
                st.info("• API rate limits - please wait a few minutes and try again")
                st.info("• Network connectivity issues")
                st.info("• Invalid API key - check your .env file")
                st.info("• Model availability issues")
                st.warning("🔄 Switching to manual analysis guidelines...")
                return "API_FAILED"
    return "API_FAILED"

def process_pdf(uploaded_file) -> tuple[str, list]:
    """
    Extract text and preview images from a PDF.
    Returns (text, preview_images). Always succeeds if possible.
    Uses OCR if pdf2image + pytesseract are available, else falls back to pdfplumber.
    """
    if not uploaded_file or uploaded_file.type != "application/pdf":
        return "", []

    bytes_data = uploaded_file.getvalue()
    text_bits, preview = [], []

    # 1️⃣ OCR route (pdf2image + pytesseract)
    if USE_PDF2IMAGE:
        try:
            images = pdf2image.convert_from_bytes(
                bytes_data,
                dpi=200,
                poppler_path=POPPLER_PATH
            )
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

            for i, img in enumerate(images):
                if i < 2:  # preview first 2 pages
                    preview.append(img)
                text_bits.append(pytesseract.image_to_string(img, config="--psm 6"))

            return "\n\n".join(text_bits), preview

        except Exception as e:
            st.warning(f"OCR pipeline failed ({e}) – falling back to text extraction")

    # 2️⃣ Text-only fallback (pdfplumber)
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(bytes_data)) as pdf:
            for i, page in enumerate(pdf.pages):
                text_bits.append(page.extract_text() or "")

                # Optional preview: only first 2 pages
                if HAS_POPPLER and i < 2:
                    try:
                        from pdf2image import convert_from_bytes
                        preview_img = convert_from_bytes(
                            bytes_data,
                            dpi=150,
                            first_page=i+1,
                            last_page=i+1,
                            poppler_path=POPPLER_PATH
                        )[0]
                        preview.append(preview_img)
                    except Exception:
                        pass  # silently skip preview if fails

        return "\n\n".join(text_bits), preview

    except Exception as e:
        st.error(f"Could not extract anything from this PDF: {e}")
        return "", []



def format_analysis_result(content: str) -> str:
    if not content:
        return "<p>No content to display.</p>"

    content = content.replace("", "")
    content = content.replace("*", "")
    content = content.replace("+", "")
    content = content.replace("•", "")
    content = content.replace("*", "")
    content = content.replace("[STRENGTH]", "")
    content = content.replace("[WEAKNESS]", "")

    sections = content.split('\n\n\n\n')
    formatted_lines = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        lines = section.split('\n')
        if len(lines) > 0:
            heading = lines[0].strip()
            if heading.endswith(':'):
                formatted_lines.append(f"{heading}")
                for line in lines[1:]:
                    line = line.strip()
                    if line:
                        formatted_lines.append(line)
            else:
                formatted_lines.append(section)

    return "<br>".join(formatted_lines)

def generate_enhanced_pdf_report(report_data: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        textColor=HexColor('#1f77b4'),
        alignment=TA_LEFT
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10,
        textColor=HexColor('#2c3e50'),
        borderWidth=1,
        borderColor=HexColor('#1f77b4'),
        borderPadding=5
    )
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_LEFT,
        spaceAfter=10,
        leftIndent=10
    )
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=20,
        bulletIndent=10,
        spaceAfter=8
    )

    elements = [
        Paragraph("📄 Resume Analysis Report", title_style),
        Spacer(1, 20)
    ]

    for section, content in report_data.items():
        if content and content.strip():
            elements.append(Paragraph(f"📋 {section}", heading_style))
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    clean_line = line.replace('[STRENGTH]', '').replace('[WEAKNESS]', '').replace("", "").strip()
                    if clean_line.startswith('•') or (len(clean_line) > 0 and clean_line[0].isdigit() and clean_line[1] == '.'):
                        elements.append(Paragraph(f"{clean_line}", bullet_style))
                    else:
                        elements.append(Paragraph(clean_line, body_style))
            elements.append(Spacer(1, 15))

    try:
        doc.build(elements)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes
    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")
        return b""

def switch_to_results_page(analysis_type: str):
    st.session_state.selected_analysis = analysis_type
    st.session_state.active_page = "Results"
    st.rerun()

# -------------------------------------
# Sidebar Navigation
# -------------------------------------
with st.sidebar:
    st.markdown("### 📌 Navigation")
    menu_options = ["Upload & Analyze", "Results", "Your AI Assistant", "Help"]
    menu = st.radio(
        "",
        menu_options,
        index=menu_options.index(st.session_state.active_page),
        key="nav_radio"
    )
    if menu != st.session_state.active_page:
        st.session_state.active_page = menu
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div class='info-box'>
    <strong>📄 Resume Recognition Expert</strong><br>
    AI-powered resume analysis and job matching system with advanced formatting and insights.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.active_page == "Upload & Analyze":
        st.markdown("""
        <div class='warning-box'>
        <strong>💡 Tips:</strong><br>
        • Upload PDF resumes only<br>
        • Ensure text is selectable<br>
        • Include detailed job descriptions
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------
# Analysis Options
# -------------------------------------
analysis_options = {
    "Quick Overview": {
        "description": "Get a rapid summary of the resume's key strengths and areas",
        "prompt": """
        Provide a comprehensive overview of this resume including key strengths and notable areas for improvement.
        Focus on the candidate's experience, skills, and overall presentation.
        """
    },
    "Issues Analysis": {
        "description": "Identify specific weaknesses and improvement areas",
        "prompt": """
        Analyze this resume and identify specific issues, weaknesses, and areas that need improvement.
        For each issue identified, mark it with [WEAKNESS] and provide specific suggestions for enhancement.
        """
    },
    "Enhancement Tips": {
        "description": "Get actionable tips to improve the resume",
        "prompt": """
        Provide specific, actionable tips to enhance this resume.
        Mark positive aspects with [STRENGTH] and improvement suggestions with [WEAKNESS].
        Include formatting, content, and presentation recommendations.
        """
    },
    "Job Matching": {
        "description": "Analyze how well the resume matches the job requirements",
        "prompt": """
        Compare this resume against the job description and analyze the match.
        Mark matching qualifications with [STRENGTH] and gaps with [WEAKNESS].
        Provide a match percentage and specific recommendations.
        """
    },
    "Complete Analysis": {
        "description": "Comprehensive analysis covering all aspects",
        "prompt": """
        Provide a complete analysis of this resume, covering all aspects including strengths, weaknesses, and job matching.
        """
    }
}

# -------------------------------------
# Main Pages
# -------------------------------------
if st.session_state.active_page == "Upload & Analyze":
    st.title("📄 Resume Recognition System")
    st.markdown("### 🚀 AI-Powered Resume Analysis & Job Matching")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="section-title">📑 Upload Resume</div>', unsafe_allow_html=True)
        uploaded_resume = st.file_uploader(
            "Choose a PDF resume file",
            type=["pdf"],
            help="Upload a PDF resume for analysis"
        )

        if uploaded_resume:
            with st.spinner("Processing resume..."):
                resume_text, resume_images = process_pdf(uploaded_resume)
                st.session_state.resume_text = resume_text
                st.session_state.preview_images = resume_images

            if resume_text:
                st.markdown('<div class="success-box">✅ Resume uploaded and processed successfully!</div>', unsafe_allow_html=True)
                word_count = len(resume_text.split())
                char_count = len(resume_text)
                col1_1, col1_2 = st.columns(2)
                with col1_1:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{word_count}</div><div class="metric-label">Words</div></div>', unsafe_allow_html=True)
                with col1_2:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{char_count}</div><div class="metric-label">Characters</div></div>', unsafe_allow_html=True)

                with st.expander("📖 View Extracted Text"):
                    st.text_area("Extracted Content", resume_text, height=200, disabled=True)
            else:
                st.error("❌ Could not extract text from the PDF. Please ensure the file contains readable text.")

    with col2:
        st.markdown('<div class="section-title">💼 Job Description</div>', unsafe_allow_html=True)
        job_description = st.text_area(
            "Paste the job description here:",
            height=300,
            placeholder="""Example:
We are seeking a Data Analyst with:
- 3+ years of experience with SQL and Python
- Experience with data visualization tools (Tableau, PowerBI)
- Strong analytical and problem-solving skills
- Bachelor's degree in related field
- Experience with statistical analysis and machine learning""",
            key="job_desc_input"
        )
        st.session_state.job_description = job_description
        if job_description:
            job_word_count = len(job_description.split())
            st.markdown(f'<div class="info-box">📊 Job Description: {job_word_count} words</div>', unsafe_allow_html=True)

    if st.session_state.preview_images:
        with st.expander("👁 Resume Preview (First 2 pages)"):
            for i, img in enumerate(st.session_state.preview_images[:2]):
                st.image(img, caption=f"Page {i+1}", use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">🔍 Choose Analysis Type</div>', unsafe_allow_html=True)
    cols = st.columns(len(analysis_options))
    for i, (analysis_type, details) in enumerate(analysis_options.items()):
        with cols[i]:
            if st.button(
                f"{analysis_type}\n\n{details['description']}",
                key=f"analysis_btn_{i}",
                use_container_width=True,
                help=f"Click to perform {analysis_type.lower()}"
            ):
                if st.session_state.resume_text and st.session_state.job_description:
                    switch_to_results_page(analysis_type)
                else:
                    if not st.session_state.resume_text:
                        st.error("❌ Please upload a resume first")
                    if not st.session_state.job_description:
                        st.error("❌ Please provide a job description")

# -------------------------------------
# Results Page
# -------------------------------------
elif st.session_state.active_page == "Results":
    st.title("📊 Analysis Results")
    if not st.session_state.resume_text:
        st.warning("⚠ No resume data found. Please upload a resume first.")
        if st.button("← Go to Upload"):
            st.session_state.active_page = "Upload & Analyze"
            st.rerun()
    elif not st.session_state.job_description:
        st.warning("⚠ No job description found. Please provide a job description.")
        if st.button("← Go to Upload"):
            st.session_state.active_page = "Upload & Analyze"
            st.rerun()
    elif st.session_state.selected_analysis:
        analysis_type = st.session_state.selected_analysis
        st.markdown(f'<div class="section-title">📋 {analysis_type}</div>', unsafe_allow_html=True)

        if analysis_type == "Complete Analysis":
            report_data = {}
            progress_bar = st.progress(0)
            status_text = st.empty()
            analysis_steps = [
                ("Quick Overview", "overview"),
                ("Issues Analysis", "issues"),
                ("Enhancement Tips", "tips"),
                ("Job Matching", "matching")
            ]

            for i, (step_name, step_key) in enumerate(analysis_steps):
                status_text.text(f"🔍 Analyzing: {step_name}...")
                progress_bar.progress((i + 1) / len(analysis_steps))
                step_options = [k for k in analysis_options.keys() if step_key in k.lower().replace(" ", "")]
                if step_options:
                    step_prompt = analysis_options[step_options[0]]["prompt"]
                    full_prompt = f"""{step_prompt}
                    Resume Content:
                    {st.session_state.resume_text[:4000]}
                    Job Description:
                    {st.session_state.job_description}
                    Please provide detailed analysis with specific examples and actionable recommendations."""
                    result = analyze_resume_with_llm(full_prompt)
                    report_data[step_name] = result
                    with st.expander(f"📋 {step_name}", expanded=True):
                        formatted_result = format_analysis_result(result)
                        st.markdown(f'<div class="analysis-result">{formatted_result}</div>', unsafe_allow_html=True)

            progress_bar.empty()
            status_text.empty()
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                pdf_bytes = generate_enhanced_pdf_report(report_data)
                if pdf_bytes:
                    st.download_button(
                        "📄 Download Complete Report",
                        pdf_bytes,
                        f"resume_analysis_report_{analysis_type.lower().replace(' ', '_')}.pdf",
                        "application/pdf",
                        use_container_width=True
                    )

        else:
            prompt_data = analysis_options.get(analysis_type, {})
            if prompt_data:
                full_prompt = f"""{prompt_data['prompt']}
                Resume Content:
                {st.session_state.resume_text[:4000]}
                Job Description:
                {st.session_state.job_description}
                Please provide detailed analysis with specific examples and actionable recommendations."""
                with st.spinner(f"🔍 Performing {analysis_type}..."):
                    result = analyze_resume_with_llm(full_prompt)

                if result == "API_FAILED":
                    st.warning("🔄 API temporarily unavailable. Showing manual analysis guidelines:")
                    manual_tips = provide_manual_analysis_tips()
                    result = manual_tips.get(analysis_type, manual_tips["Quick Overview"])

                formatted_result = format_analysis_result(result)
                st.markdown(f'<div class="analysis-result">{formatted_result}</div>', unsafe_allow_html=True)

                if result == "API_FAILED" or "API temporarily unavailable" in result:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("🔄 Retry AI Analysis", use_container_width=True, type="primary"):
                            st.rerun()

                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    pdf_bytes = generate_enhanced_pdf_report({analysis_type: result})
                    if pdf_bytes:
                        st.download_button(
                            f"📄 Download {analysis_type} Report",
                            pdf_bytes,
                            f"resume_analysis_{analysis_type.lower().replace(' ', '_')}.pdf",
                            "application/pdf",
                            use_container_width=True
                        )

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Upload", use_container_width=True):
                st.session_state.active_page = "Upload & Analyze"
                st.rerun()
        with col2:
            if st.button("💬 Ask AI Assistant", use_container_width=True):
                st.session_state.active_page = "Your AI Assistant"
                st.rerun()

    else:
        st.info("👆 No analysis type selected. Please go back and choose an analysis type.")
        if st.button("← Go to Upload"):
            st.session_state.active_page = "Upload & Analyze"
            st.rerun()

# -------------------------------------
# AI Assistant Page
# -------------------------------------
elif st.session_state.active_page == "Your AI Assistant":
    st.title("🤖 Your AI Assistant")
    st.markdown("### Ask specific questions about your resume")

    if not st.session_state.resume_text:
        st.warning("⚠ Please upload a resume first to use the AI assistant.")
        if st.button("← Go to Upload"):
            st.session_state.active_page = "Upload & Analyze"
            st.rerun()
    else:
        st.markdown("#### 💡 Quick Questions")
        quick_questions = [
            "What are the strongest points of my resume?",
            "What skills should I highlight more?",
            "How can I improve my resume summary?",
            "What experience should I emphasize?",
            "Are there any red flags in my resume?"
        ]
        cols = st.columns(3)
        for i, question in enumerate(quick_questions):
            with cols[i % 3]:
                if st.button(question, key=f"quick_q_{i}", use_container_width=True):
                    st.session_state.current_question = question

        st.markdown("---")
        col1, col2 = st.columns([4, 1])
        with col1:
            user_question = st.text_input(
                "💬 Ask your question:",
                value=st.session_state.get('current_question', ''),
                key="chat_input_field",
                placeholder="e.g., How can I better highlight my leadership experience?"
            )
        with col2:
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                ask_button = st.button("Ask", use_container_width=True, type="primary")
            with col2_2:
                if st.button("Clear", use_container_width=True):
                    st.session_state.chat_history = []
                    st.session_state.current_question = ""
                    st.rerun()

        if (ask_button or st.session_state.get('current_question')) and user_question.strip():
            question_to_process = user_question or st.session_state.get('current_question', '')
            if question_to_process:
                prompt = f"""
You are a professional resume consultant. Answer the user's question based ONLY on the following resume content.
Be specific, actionable, and helpful. Use examples from the resume when possible.
Resume Content:
{st.session_state.resume_text[:4000]}
Job Context (if provided):
{st.session_state.job_description[:1000] if st.session_state.job_description else "No specific job provided"}
User Question: {question_to_process}
Provide a detailed, professional response with specific recommendations.
"""
                with st.spinner("🤔 Analyzing your question..."):
                    answer = analyze_resume_with_llm(prompt)

                st.session_state.chat_history.append({
                    "role": "user",
                    "content": question_to_process
                })
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })
                if 'current_question' in st.session_state:
                    del st.session_state.current_question
                st.rerun()

        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("#### 💬 Conversation History")
            for msg in reversed(st.session_state.chat_history[-10:]):
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div style='background: linear-gradient(90deg, #1f77b4, #1565C0); padding: 15px; border-radius: 10px; margin: 10px 0;'>
                    <strong>👤 You:</strong> {msg['content'].replace("", "")}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    formatted_answer = format_analysis_result(msg['content'].replace("", ""))
                    st.markdown(f"""
                    <div class='analysis-result'>
                    <strong>🤖 AI Assistant:</strong><br><br>
                    {formatted_answer}
                    </div>
                    """, unsafe_allow_html=True)

# -------------------------------------
# Help Page
# -------------------------------------
elif st.session_state.active_page == "Help":
    st.title("📚 User Guide")
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Getting Started", "🔧 Features", "💡 Tips", "❓ FAQ"])

    with tab1:
        st.markdown("""
        ### How to Use the Resume Recognition System
        <div class='info-box'>
        <strong>Step 1: Upload Your Resume</strong><br>
        • Go to "Upload & Analyze" page<br>
        • Choose a PDF file of your resume<br>
        • Wait for text extraction to complete
        </div>
        <div class='info-box'>
        <strong>Step 2: Add Job Description</strong><br>
        • Paste the job posting you're interested in<br>
        • Include requirements, responsibilities, and qualifications<br>
        • The more detailed, the better the analysis
        </div>
        <div class='info-box'>
        <strong>Step 3: Choose Analysis Type</strong><br>
        • <strong>Quick Overview:</strong> Fast summary of key points<br>
        • <strong>Issues Analysis:</strong> Identify problems and gaps<br>
        • <strong>Enhancement Tips:</strong> Get improvement suggestions<br>
        • <strong>Job Matching:</strong> See how well you match the job<br>
        • <strong>Complete Analysis:</strong> Full comprehensive review
        </div>
        <div class='info-box'>
        <strong>Step 4: Review Results</strong><br>
        • Results automatically open when analysis starts<br>
        • Green checkmarks (✓) show strengths<br>
        • Red X marks (✗) show areas for improvement<br>
        • Download PDF reports for your records
        </div>
        <div class='info-box'>
        <strong>Step 5: Ask Follow-up Questions</strong><br>
        • Use "Your AI Assistant" for specific questions<br>
        • Get personalized advice about your resume<br>
        • Ask for clarifications or additional tips
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        ### 🎯 Key Features
        #### Resume Analysis Types
        - Quick Overview: 2-minute summary of your resume's key points
        - Issues Analysis: Detailed identification of problems and gaps
        - Enhancement Tips: Actionable suggestions for improvement
        - Job Matching: Compatibility analysis with specific job requirements
        - Complete Analysis: Comprehensive review covering all aspects
        #### Smart Features
        - OCR Text Extraction: Automatically reads text from PDF resumes
        - Color-Coded Results: Green for strengths, red for weaknesses
        - PDF Report Generation: Download professional analysis reports
        - Interactive AI Assistant: Ask specific questions about your resume
        - Resume Preview: Visual preview of uploaded documents
        #### Analysis Categories
        - Content quality and relevance
        - Formatting and presentation
        - Skills alignment with job requirements
        - Experience relevance and presentation
        - Overall professional impression
        """)

    with tab3:
        st.markdown("""
        ### 💡 Tips for Best Results
        <div class='success-box'>
        <strong>📄 Resume Upload Tips</strong><br>
        • Use high-quality PDF files<br>
        • Ensure text is selectable (not scanned images)<br>
        • Keep file size under 10MB<br>
        • Use standard fonts and formatting
        </div>
        <div class='success-box'>
        <strong>💼 Job Description Tips</strong><br>
        • Include complete job postings, not just titles<br>
        • Copy requirements, responsibilities, and qualifications<br>
        • Include company information if available<br>
        • The more detail, the better the matching analysis
        </div>
        <div class='success-box'>
        <strong>🎯 Getting Better Analysis</strong><br>
        • Start with "Complete Analysis" for comprehensive insights<br>
        • Use specific analysis types for targeted feedback<br>
        • Ask follow-up questions in the AI Assistant<br>
        • Download reports for future reference
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.markdown("""
        ### ❓ Frequently Asked Questions
        Q: What file formats are supported?
        A: Currently, only PDF files are supported. Make sure your PDF contains selectable text.
        Q: Why can't the system read my resume?
        A: This usually happens with scanned PDFs or image-based files. Try using a PDF with selectable text.
        Q: How accurate is the job matching analysis?
        A: The system uses advanced AI to analyze compatibility, but results should be used as guidance alongside your professional judgment.
        Q: Can I save my analysis results?
        A: Yes! Use the download button to save PDF reports of your analysis results.
        Q: Is my resume data stored anywhere?
        A: No, your resume data is only processed during your session and is not permanently stored.
        Q: What if the analysis seems incorrect?
        A: Use the AI Assistant to ask for clarifications or different perspectives on specific points.
        Q: Can I analyze multiple resumes?
        A: You can upload different resumes in the same session, but each upload will replace the previous one.
        """)

# -------------------------------------
# Footer
# -------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
<small>🤖 Powered by AI • Built with Streamlit • Resume Recognition System v2.0</small>
</div>
""", unsafe_allow_html=True)

