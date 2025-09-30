"""
Streamlit UI for CV Creator Capstone

This single-file Streamlit app provides an interactive front-end for the
CV Creation prototype. It wraps the extended pipeline that integrates:
 - Gemma 3 1B via Ollama (tailoring)
 - LlamaIndex (parsing)
 - LangChain (orchestration)
 - ResumeLM (optional generator/scorer)

Usage:
  export OLLAMA_URL="http://localhost:11434"
  streamlit run cv_creator_streamlit.py

Notes:
 - This file intentionally keeps the core pipeline functions local so the app is runnable standalone.
 - If you already have the prototype file, you can import from it instead of duplicating code.
 - ResumeLM is optional - the app works fully without it using built-in ATS scoring.

Requirements:
  pip install streamlit pdfplumber python-docx langchain llama-index requests
  (Clone ResumeLM repo and add to PYTHONPATH if you want optional support.)

"""

import streamlit as st
import tempfile
import os
import json
import re
from pathlib import Path

# ---- LLM + libs ----
try:
    import pdfplumber
    from docx import Document
    import requests
    
    # Updated imports for newer versions
    try:
        from langchain_community.llms import Ollama
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        print("⚠️ LangChain not available")
        Ollama = None
        PromptTemplate = None
        LLMChain = None
        LANGCHAIN_AVAILABLE = False
    
    try:
        from llama_index.core import Document as LIDoc
        from llama_index.core import VectorStoreIndex
        from llama_index.core import Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        # Configure LlamaIndex to use local embeddings
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        LLAMAINDEX_AVAILABLE = True
    except ImportError:
        print("⚠️ LlamaIndex not available")
        LIDoc = None
        VectorStoreIndex = None
        LLAMAINDEX_AVAILABLE = False
        
except Exception as e:
    st.error(f"Missing one or more dependencies: {e}")
    raise

# Optional ResumeLM
try:
    from resumelm import ResumeBuilder, ATSScorer
    HAS_RESUMELM = True
except Exception:
    ResumeBuilder = None
    ATSScorer = None
    HAS_RESUMELM = False

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:1b")

# Minimal Ollama wrapper if langchain Ollama unavailable
def find_working_ollama_url():
    """Find a working Ollama URL from common options"""
    test_urls = [
        "http://localhost:11434",
        "http://127.0.0.1:11434", 
        "http://0.0.0.0:11434"
    ]
    
    for url in test_urls:
        try:
            response = requests.get(f"{url}/api/version", timeout=2)
            if response.status_code == 200:
                return url
        except:
            continue
    
    return OLLAMA_URL  # Return default as fallback

def call_ollama_direct(prompt: str, model: str = OLLAMA_MODEL, temperature: float = 0.0, max_tokens: int = 1024):
    # Try to find working URL dynamically
    working_url = find_working_ollama_url()
    url = f"{working_url}/api/generate"
    
    payload = {
        "model": model, 
        "prompt": prompt, 
        "stream": False,  # Important: disable streaming for simpler handling
        "options": {
            "temperature": temperature, 
            "num_predict": max_tokens
        }
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        
        # Handle Ollama's response format
        if isinstance(data, dict) and 'response' in data:
            return data['response']
        elif isinstance(data, dict):
            # Try other possible response fields
            if 'results' in data and isinstance(data['results'], list):
                return data['results'][0].get('content', '')
            if 'text' in data:
                return data['text']
            if 'result' in data:
                return data['result']
        
        return str(data)
        
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Please make sure Ollama is running and accessible."
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The model might be taking too long to respond."
    except Exception as e:
        return f"Error calling Ollama: {e}"

# Utilities

def extract_text_from_pdf(path: str) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            texts.append(p.extract_text() or "")
    return "\n\n".join(texts)


def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

# LlamaIndex small wrapper with error handling
def parse_with_llama_index(text: str, instruction: str) -> str:
    if not LLAMAINDEX_AVAILABLE or LIDoc is None or VectorStoreIndex is None:
        # Fallback to simple text processing
        return simple_text_extraction(text, instruction)
    
    try:
        doc = LIDoc(text=text)
        index = VectorStoreIndex.from_documents([doc])
        qe = index.as_query_engine()
        r = qe.query(instruction)
        return getattr(r, 'response', str(r))
    except Exception as e:
        print(f"⚠️ LlamaIndex failed: {e}")
        return simple_text_extraction(text, instruction)

def simple_text_extraction(text: str, instruction: str) -> str:
    """Simple fallback text extraction when LlamaIndex fails"""
    if "name" in instruction.lower() and "contact" in instruction.lower():
        # Extract first few lines for contact info
        lines = text.split('\n')[:5]
        return '\n'.join([line.strip() for line in lines if line.strip()])
    elif "skills" in instruction.lower():
        # Look for skills
        skill_keywords = ['python', 'java', 'javascript', 'react', 'sql', 'html', 'css', 'git']
        found_skills = []
        for line in text.split('\n'):
            for skill in skill_keywords:
                if skill.lower() in line.lower():
                    found_skills.append(skill.title())
        return f'{{"skills": {list(set(found_skills))}}}'
    else:
        # Return first 500 characters
        return text[:500]

# Force disable LangChain to avoid connection issues
USE_LANGCHAIN = False
resume_chain = None

print("🔧 LangChain disabled - using direct Ollama API only")

# Simple safe JSON extractor
import re, json

def safe_json_from_text(text: str):
    if not isinstance(text, str):
        return text
    m = re.search(r"(\{[\s\S]*\})", text)
    if m:
        blob = m.group(1)
        try:
            return json.loads(blob)
        except Exception:
            try:
                return json.loads(blob.replace("'", '"'))
            except Exception:
                return blob
    # fallback
    return text

# ATS keyword score

def ats_keyword_score(resume_text: str, keywords: list):
    if not keywords:
        return {"matched": [], "score": 0.0}
    rl = resume_text.lower()
    found = [k for k in keywords if k.lower() in rl]
    score = len(found)/len(keywords)
    return {"matched": found, "score": round(score,3)}

# Streamlit UI
st.title("CV Creator Capstone — Streamlit UI")
st.markdown("Upload a resume (PDF/DOCX) and a job description (TXT). The app will parse, tailor, and output a resume.")

# Test Ollama connection
with st.sidebar:
    st.header("🔧 System Status")
    
    # Test Ollama connection with multiple URLs
    working_url = find_working_ollama_url()
    
    try:
        test_response = requests.get(f"{working_url}/api/version", timeout=5)
        if test_response.status_code == 200:
            version_data = test_response.json()
            st.success("✅ Ollama Connected")
            st.write(f"**URL:** {working_url}")
            st.write(f"**Version:** {version_data.get('version', 'Unknown')}")
            st.write(f"**Model:** {OLLAMA_MODEL}")
            
            # Test if model is available
            models_response = requests.get(f"{working_url}/api/tags", timeout=5)
            if models_response.status_code == 200:
                models = models_response.json()
                model_names = [m.get('name', '') for m in models.get('models', [])]
                if OLLAMA_MODEL in model_names:
                    st.success(f"✅ Model {OLLAMA_MODEL} available")
                else:
                    st.warning(f"⚠️ Model {OLLAMA_MODEL} not found")
                    st.write("Available models:", model_names)
        else:
            st.error("❌ Ollama Not Responding")
    except Exception as e:
        st.error(f"❌ Ollama Connection Failed: {e}")
        st.warning("The app will not work without Ollama running")
        st.markdown("""
        **Troubleshooting:**
        1. Start Ollama: `ollama serve`
        2. Set host binding: `set OLLAMA_HOST=0.0.0.0:11434`
        3. Restart Ollama service
        """)
    
    # Show available features
    st.write("**Available Features:**")
    st.write(f"• LangChain: {'✅' if LANGCHAIN_AVAILABLE else '❌'}")
    st.write(f"• LlamaIndex: {'✅' if LLAMAINDEX_AVAILABLE else '❌'}")
    st.write(f"• ResumeLM (Optional): {'✅' if HAS_RESUMELM else '➖'}")
    if not HAS_RESUMELM:
        st.caption("💡 ResumeLM is optional - core functionality works without it")

uploaded_resume = st.file_uploader("Upload resume (PDF or DOCX)", type=["pdf","docx"] )
uploaded_job = st.file_uploader("Upload job description (TXT)", type=["txt"]) 

if uploaded_resume and uploaded_job:
    with st.spinner('Processing...'):
        # save temp files
        tdir = tempfile.mkdtemp()
        resume_path = os.path.join(tdir, uploaded_resume.name)
        job_path = os.path.join(tdir, uploaded_job.name)
        with open(resume_path, 'wb') as f:
            f.write(uploaded_resume.getbuffer())
        with open(job_path, 'wb') as f:
            f.write(uploaded_job.getbuffer())

        # extract text
        if uploaded_resume.name.lower().endswith('.pdf'):
            resume_text = extract_text_from_pdf(resume_path)
        else:
            resume_text = extract_text_from_docx(resume_path)
        job_text = Path(job_path).read_text(encoding='utf-8')

        # parse with LlamaIndex
        inst_resume = "Extract name, contact, skills, education, work experience, projects in JSON."
        inst_job = "Extract title, responsibilities, requirements, keywords in JSON."
        parsed_resume = parse_with_llama_index(resume_text, inst_resume)
        parsed_job = parse_with_llama_index(job_text, inst_job)

        st.subheader('Parsed Resume (raw from LlamaIndex)')
        st.code(parsed_resume[:2000] + ("..." if len(parsed_resume)>2000 else ""))
        st.subheader('Parsed Job (raw from LlamaIndex)')
        st.code(parsed_job[:2000] + ("..." if len(parsed_job)>2000 else ""))

        # Tailor with direct Ollama API (more reliable)
        st.info("🔄 Using direct Ollama API for tailoring...")
        prompt = f"""You are an expert CV writer. Create a tailored resume based on the following:

PARSED RESUME DATA:
{parsed_resume}

JOB REQUIREMENTS:
{parsed_job}

Please create a professional, ATS-friendly resume in markdown format that:
1. Highlights skills matching the job requirements
2. Emphasizes relevant experience
3. Uses keywords from the job description
4. Maintains a clean, professional structure

Return only the markdown resume without any additional commentary."""

        tailored = call_ollama_direct(prompt, model=OLLAMA_MODEL, temperature=0.2, max_tokens=2000)

        st.subheader('Tailored Resume (Markdown)')
        st.text_area('Tailored Resume', value=tailored, height=300)

        # optional ResumeLM
        if HAS_RESUMELM:
            try:
                builder = ResumeBuilder()
                final_resume = builder.build(parsed_resume, parsed_job)
                st.subheader('ResumeLM output (JSON)')
                st.json(final_resume)
                if ATSScorer:
                    scorer = ATSScorer()
                    score = scorer.score(json.dumps(final_resume), job_text)
                    st.write('ResumeLM ATS score:', score)
            except Exception as e:
                st.warning('ResumeLM present but failed: ' + str(e))

        # ATS keyword scoring quick check
        # extract keywords from parsed_job (attempt JSON)
        kj = []
        pj = safe_json_from_text(parsed_job)
        if isinstance(pj, dict):
            kj = pj.get('keywords') or pj.get('requirements') or []
            if isinstance(kj, str):
                kj = [k.strip() for k in re.split(r'[,;]|\n', kj) if k.strip()]
        else:
            kj = []

        score = ats_keyword_score(tailored, kj)
        st.subheader('Quick ATS Keyword Score')
        st.write(score)

        # Download buttons
        st.download_button('Download Markdown', tailored, file_name='tailored_resume.md', mime='text/markdown')
        # also provide JSON download
        try:
            md_json = {'tailored_markdown': tailored, 'parsed_resume': parsed_resume, 'parsed_job': parsed_job}
            st.download_button('Download Result JSON', json.dumps(md_json, indent=2), file_name='tailored_result.json', mime='application/json')
        except Exception:
            pass

        st.success('Done — check the outputs above.')
else:
    st.info('Upload both files to begin.')
