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
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma-3-1b")

# Minimal Ollama wrapper if langchain Ollama unavailable
def call_ollama_direct(prompt: str, model: str = OLLAMA_MODEL, temperature: float = 0.0, max_tokens: int = 1024):
    url = os.environ.get('OLLAMA_URL', 'http://localhost:11434') + '/api/generate'
    payload = {"model": model, "prompt": prompt, "temperature": temperature, "max_tokens": max_tokens}
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # flexible parsing
    if isinstance(data, dict):
        if 'results' in data and isinstance(data['results'], list):
            return data['results'][0].get('content', '')
        if 'text' in data:
            return data['text']
        if 'result' in data:
            return data['result']
    return str(data)

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

# LangChain + Ollama tailoring chain (fallback to direct call if unavailable)
try:
    if LANGCHAIN_AVAILABLE and Ollama is not None:
        ollama_llm = Ollama(model=OLLAMA_MODEL)
        resume_prompt = PromptTemplate(
            input_variables=["resume_json", "job_json"],
            template=("You are an expert CV writer. Given resume JSON and job JSON, return a polished resume in markdown.\n"
                     "Resume: {resume_json}\nJob: {job_json}\n")
        )
        resume_chain = LLMChain(llm=ollama_llm, prompt=resume_prompt)
        USE_LANGCHAIN = True
    else:
        resume_chain = None
        USE_LANGCHAIN = False
except Exception as e:
    print(f"⚠️ LangChain setup failed: {e}")
    resume_chain = None
    USE_LANGCHAIN = False

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

        # Tailor with LangChain or direct Ollama
        if USE_LANGCHAIN and resume_chain:
            tailored = resume_chain.run(resume_json=parsed_resume, job_json=parsed_job)
        else:
            # fallback prompt for direct call
            prompt = f"Tailor this resume JSON to this job JSON and return markdown resume:\nResume: {parsed_resume}\nJob: {parsed_job}\n"
            tailored = call_ollama_direct(prompt, model=OLLAMA_MODEL, temperature=0.2, max_tokens=1500)

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
