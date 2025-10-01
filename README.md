# CV Creation using LLMs — Capstone Project

This project demonstrates how modern open-source LLMs and frameworks can be used to **automate the creation of tailored CVs** from resumes and job descriptions. It is designed as a **capstone prototype** showcasing integration of multiple OSS tools.

---

## 🚀 Features

- **Gemma 3 1B via Ollama** — lightweight, local LLM for tailoring resumes.
- **LlamaIndex** — parses resumes and job descriptions into structured JSON.
- **LangChain** — orchestrates multi-step LLM pipelines.
- **ResumeLM** — optional ATS-optimized resume builder and scorer.
- **Streamlit UI** — simple interactive web app for uploading resumes and job descriptions.
- **CLI mode** — build tailored CVs from the terminal.
- **ATS Keyword Scoring** — quick heuristic score of resume vs job keywords.

---

## 📂 Project Structure

```
cv_creator_capstone_extended.py   # Extended CLI prototype (Gemma + LlamaIndex + LangChain + ResumeLM)
cv_creator_streamlit.py           # Streamlit UI for interactive demo
README.md                         # Documentation (this file)
```

---

## ⚙️ Installation

1. Clone this repository and navigate into the project folder.
2. Install Python dependencies:
   ```bash
   pip install streamlit pdfplumber python-docx langchain llama-index requests
   ```
3. (Optional) Install **ResumeLM**:
   ```bash
   git clone https://github.com/sudip2k17/cv-creator-with-llm.git
   cd resumelm && pip install -e .
   ```
4. Ensure you have **Ollama** installed and running with **Gemma 3** model:
   ```bash
   ollama run gemma:3b
   ```

---

## ▶️ Usage

### 1. CLI Mode

Run the extended prototype directly:

```bash
python cv_creator_capstone_extended.py --resume sample_resume.pdf --job job.txt --out tailored_resume.json
```

### 2. Streamlit UI

Launch the interactive web interface:

```bash
streamlit run cv_creator_streamlit.py
```

Upload a resume (PDF/DOCX) and a job description (TXT), and download the tailored CV.

---

## 📊 Workflow

1. **Upload/Parse Resume** → Extract structured fields with **LlamaIndex**.
2. **Upload/Parse Job Description** → Extract role responsibilities, skills, and keywords.
3. **Tailor Resume** → Use **Gemma (via Ollama + LangChain)** to rewrite and optimize.
4. **Optional ResumeLM Step** → Generate ATS-optimized resume and score.
5. **Review & Export** → Download as Markdown/JSON/DOCX.

---

## 🧩 Capstone Objectives

- Showcase integration of multiple OSS LLM frameworks.
- Demonstrate practical **NLP + document automation**.
- Provide both a research prototype (CLI) and a demo-ready interface (Streamlit).

---

## 🔮 Future Enhancements

- Add **DOCX/PDF export** in the Streamlit UI.
- Enhance **ATS scoring** using embeddings (semantic similarity).
- Provide a **pre-seeded demo dataset** (sample resumes & jobs).
- Deploy Streamlit app online (e.g., Streamlit Cloud, Hugging Face Spaces).

---

## 👨‍💻 Authors

- Sudip Sengupta

---

## 📜 License

This project is for educational/capstone purposes. Integrations with Ollama, LlamaIndex, LangChain, and ResumeLM respect their individual licenses.

