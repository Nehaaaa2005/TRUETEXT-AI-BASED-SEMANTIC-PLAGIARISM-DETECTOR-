from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from typing import List
import os
import re
import numpy as np
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import Form
from datetime import datetime
from starlette.middleware.sessions import SessionMiddleware
from fastapi import Request
import wikipedia
from difflib import SequenceMatcher
import sqlite3
import bcrypt
from pydantic import BaseModel

app = FastAPI(title="TrueText AI", version="3.0")

app.add_middleware(SessionMiddleware, secret_key="supersecretkey")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")
from transformers import pipeline

ai_detector = pipeline(
    "text-classification",
    model="roberta-base-openai-detector"
) 


# =======================
# HELPER FUNCTIONS
# =======================

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def clean_text(text):
    text = re.sub(r"\s+", " ", text)   # normalize spaces
    text = re.sub(r"[^a-zA-Z0-9., ]", "", text)
    return text.strip().lower()

def chunk_text(text, chunk_size=80):
    words = text.split(" ")
    return [
        " ".join(words[i:i+chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

# =====================================================
# EXACT TEXT MATCHING (for copy-paste detection)
# =====================================================

def exact_match_score(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()


# =====================================================
# SMART WIKIPEDIA RETRIEVAL ENGINE
# =====================================================

def get_wikipedia_pages(query_list):
    pages = []
    
    for query in query_list:
        try:
            search_results = wikipedia.search(query)
            
            for title in search_results[:2]:
                try:
                    page = wikipedia.page(title, auto_suggest=True)
                    pages.append({
                        "title": page.title,
                        "content": page.content,
                        "url": page.url
                    })
                except:
                    continue
        except:
            continue
    
    return pages

import textstat

def analyze_ai_likelihood(text):

    # Sentence statistics
    sentences = text.split(".")
    words = text.split()

    if len(sentences) < 2 or len(words) < 30:
        return "Text too short for analysis", 0

    sentence_lengths = [len(s.split()) for s in sentences if len(s.split()) > 0]

    avg_len = np.mean(sentence_lengths)
    variation = np.std(sentence_lengths)

    # Repetition score
    unique_words = len(set(words))
    repetition_ratio = 1 - (unique_words / len(words))

    # Readability score
    readability = textstat.flesch_reading_ease(text)

    # Heuristic AI scoring
    ai_score = 0

    if avg_len > 18:
        ai_score += 25
    if variation < 5:
        ai_score += 25
    if repetition_ratio > 0.35:
        ai_score += 25
    if readability < 40:
        ai_score += 25

    ai_score = min(ai_score, 100)

    # Verdicts
    if ai_score > 70:
        verdict = "Highly Likely AI-Generated"
    elif ai_score > 40:
        verdict = "Possibly AI-Assisted"
    else:
        verdict = "Likely Human-Written"

    return verdict, ai_score

# =======================
# SPA ROUTE
# =======================

@app.get("/")
def serve_spa():
    return FileResponse("../frontend/index.html")

# =======================
# SINGLE FILE vs CORPUS
# =======================

@app.post("/check-single-file")
async def check_single_file(
    request: Request,
    file: UploadFile = File(None),
    text: str = Form(None)
):

    if not request.session.get("admin"):
        raise HTTPException(status_code=403, detail="Admin login required")

   

    # -------------------
    # INPUT HANDLING
    # -------------------

    if file:

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        if file.filename.lower().endswith(".pdf"):
            raw_text = extract_text_from_pdf(file_path)
        elif file.filename.lower().endswith(".docx"):
            raw_text = extract_text_from_docx(file_path)
        else:
            raw_text = ""

    elif text:
        raw_text = text

    else:
        raise HTTPException(status_code=400, detail="No input provided")

    # -------------------
    # WEB-ONLY PROCESSING
    # -------------------

    cleaned_file = clean_text(raw_text)
    file_chunks = chunk_text(cleaned_file)

    # Ignore very short content
    if len(cleaned_file.split()) < 40:
        return {
            "plagiarism_percentage": 0,
            "matched_chunks": [],
            "report_url": None
        }

    file_embeddings = model.encode(file_chunks)

    # =========================
    # SMART WIKIPEDIA WEB CHECK
    # =========================

    # Extract topic keywords
    words = raw_text.split()
    topic_words = [w for w in words if w[0].isupper()]

    queries = []

    if topic_words:
        queries.append(" ".join(topic_words[:2]))
        queries.append(" ".join(topic_words[:3]))

    queries.append(" ".join(cleaned_file.split()[:6]))
    queries.append(" ".join(cleaned_file.split()[:10]))

    # Fetch multiple Wikipedia pages
    wiki_pages = get_wikipedia_pages(queries)

    all_wiki_text = ""
    source_links = []

    for page in wiki_pages:
        all_wiki_text += page["content"]
        source_links.append((page["title"], page["url"]))

    cleaned_wiki = clean_text(all_wiki_text)
    wiki_chunks = chunk_text(cleaned_wiki)

    matched_chunks = []

    if wiki_chunks:
        wiki_embeddings = model.encode(wiki_chunks)
        wiki_similarity_matrix = cosine_similarity(file_embeddings, wiki_embeddings)
        wiki_max_sim = wiki_similarity_matrix.max(axis=1)

        # Average semantic similarity
        semantic_score = np.mean(wiki_max_sim) * 100

        # Strong match ratio (bonus weight)
        strong_ratio = (np.sum(wiki_max_sim >= 0.65) / len(wiki_max_sim)) * 100

        # Combined weighted score
        plagiarism_score = (0.85 * semantic_score) + (0.15 * strong_ratio)

        # =========================
        # EXACT MATCH BOOSTER
        # =========================

        exact_scores = []

        for fchunk in file_chunks:
            chunk_best = 0
            for wchunk in wiki_chunks:
                score = exact_match_score(fchunk, wchunk)
                chunk_best = max(chunk_best, score)
            exact_scores.append(chunk_best)

        exact_score = np.mean(exact_scores) * 100
        
        # Take higher of semantic or exact score
        plagiarism_score = max(plagiarism_score, exact_score)

        if len(wiki_max_sim) > 0:
            highlight_threshold = np.percentile(wiki_max_sim, 70)
        else:
            highlight_threshold = 0.6

        matched_indices = np.where(wiki_max_sim >= highlight_threshold)[0]
        matched_chunks = [file_chunks[i] for i in matched_indices]
    else:
        plagiarism_score = 0
        source_links = []

    # =====================================================
    # REPORT GENERATION & RESULT PACKAGING
    # =====================================================

    # Initialize highlighted text safely
    highlighted_text = cleaned_file if cleaned_file else "No text available"

    # Highlight matches
    for chunk in set(matched_chunks):
        highlighted_text = highlighted_text.replace(
            chunk,
            f"<mark>{chunk}</mark>"
        )

    current_time = datetime.now().strftime("%d %B %Y | %H:%M")

    if plagiarism_score >= 45:
        verdict = "High Similarity"
    elif plagiarism_score >= 20:
        verdict = "Moderate Similarity"
    else:
        verdict = "Low Similarity"

    # Format source links
    sources_html = ""
    for title, url in source_links:
        sources_html += f"<li><a href='{url}' target='_blank'>{title}</a></li>"

    report_content = f"""
    <html>
    <head>
    <title>TrueText AI Report</title>
    <style>
    body {{
        font-family: 'Segoe UI', sans-serif;
        background: #eef7ff;
        padding: 40px;
    }}
    .container {{
        background: white;
        padding: 40px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    }}
    .header {{
        border-bottom: 2px solid #0077b6;
        margin-bottom: 20px;
    }}
    .title {{
        color: #0077b6;
        font-size: 28px;
        font-weight: bold;
    }}
    .meta {{
        color: gray;
        font-size: 14px;
    }}
    .score-box {{
        background: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }}
    .score {{
        font-size: 32px;
        font-weight: bold;
        color: #0077b6;
    }}
    .verdict {{
        font-size: 18px;
        font-weight: 600;
        margin-top: 5px;
    }}
    .section {{
        margin-top: 30px;
    }}
    mark {{
        background-color: #fff176;
    }}
    .footer {{
        margin-top: 40px;
        font-size: 12px;
        text-align: center;
        color: gray;
    }}
    </style>
    </head>
    <body>

    <div class="container">

    <div class="header">
    <div class="title">TrueText AI Plagiarism Report</div>
    <div class="meta">Generated on: {current_time}</div>
    </div>

    <div class="score-box">
    <div class="score">{round(plagiarism_score,2)}%</div>
    <div class="verdict">{verdict}</div>
    </div>

    <div class="section">
    <h3>Highlighted Similar Content</h3>
    <p>{highlighted_text}</p>
    </div>

    <div class="section">
    <h3>Sources Referenced</h3>
    <ul>
    {sources_html}
    </ul>
    </div>

    <div class="footer">
    TrueText AI | Semantic Plagiarism Detection using SBERT Embeddings  
    © 2026 Neha Suresh.
    </div>

    </div>

    </body>
    </html>
    """

    report_path = os.path.join(UPLOAD_DIR, "report.html")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return {
        "plagiarism_percentage": round(plagiarism_score, 2),
        "matched_chunks": matched_chunks,
        "report_url": "/download-report"
    }

@app.post("/check-plagiarism")
async def check_plagiarism(request: Request, files: List[UploadFile] = File(...)):

    if not (request.session.get("admin") or request.session.get("user")):
        raise HTTPException(status_code=403, detail="Admin login required")

    if len(files) != 2:
        raise HTTPException(status_code=400, detail="Upload exactly 2 files")

    texts = []

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        if file.filename.lower().endswith(".pdf"):
            raw_text = extract_text_from_pdf(file_path)
        elif file.filename.lower().endswith(".docx"):
            raw_text = extract_text_from_docx(file_path)
        else:
            raw_text = ""

        texts.append(clean_text(raw_text))

    chunks1 = chunk_text(texts[0])
    chunks2 = chunk_text(texts[1])

    emb1 = model.encode(chunks1)
    emb2 = model.encode(chunks2)

    sim_matrix = cosine_similarity(emb1, emb2)
    max_sim = sim_matrix.max(axis=1)

    plagiarism_score = float(np.mean(max_sim) * 100)

    return {
        "plagiarism_percentage": round(plagiarism_score, 2)
    }

class TextInput(BaseModel):
    text1: str
    text2: str

@app.post("/detect-ai")
async def detect_ai(request: Request, text: str = Form(...)):

    if not (request.session.get("admin") or request.session.get("user")):
        raise HTTPException(status_code=403, detail="Admin login required")

    # -------------------------
    # CLEAN TEXT
    # -------------------------
    cleaned = clean_text(text)

    # -------------------------
    # 1. RoBERTa Prediction
    # -------------------------
    try:
        result = ai_detector(text)[0]
        roberta_label = result['label']
        roberta_score = result['score'] * 100

        if roberta_label == "LABEL_1":
            roberta_verdict = "AI"
        else:
            roberta_verdict = "HUMAN"

    except:
        roberta_verdict = "UNKNOWN"
        roberta_score = 0

    # -------------------------
    # 2. Heuristic Prediction
    # -------------------------
    heuristic_verdict, heuristic_score = analyze_ai_likelihood(cleaned)

    if "Highly" in heuristic_verdict:
        heuristic_label = "AI"
    elif "Possibly" in heuristic_verdict:
        heuristic_label = "AI"
    else:
        heuristic_label = "HUMAN"

    # -------------------------
    # 3. FINAL DECISION (Hybrid)
    # -------------------------
    if roberta_verdict == "AI" and heuristic_label == "AI":
        final_verdict = "Highly Likely AI-Generated"
        confidence = (roberta_score + heuristic_score) / 2

    elif roberta_verdict == "AI" or heuristic_label == "AI":
        final_verdict = "Possibly AI-Assisted"
        confidence = max(roberta_score, heuristic_score)

    else:
        final_verdict = "Likely Human-Written"
        confidence = min(roberta_score, heuristic_score)

    # -------------------------
    # RETURN RESULT
    # -------------------------
    return {
        "ai_verdict": final_verdict,
        "confidence_percentage": round(confidence, 2),
        "roberta_score": round(roberta_score, 2),
        "heuristic_score": round(heuristic_score, 2)
    }

@app.post("/check-text")
async def check_text(request: Request, data: TextInput):

    if not (request.session.get("admin") or request.session.get("user")):
        raise HTTPException(status_code=403, detail="Admin login required")

    text1 = clean_text(data.text1)
    text2 = clean_text(data.text2)
    
    chunks1 = chunk_text(text1)
    chunks2 = chunk_text(text2)

    emb1 = model.encode(chunks1)
    emb2 = model.encode(chunks2)

    sim_matrix = cosine_similarity(emb1, emb2)
    max_sim = sim_matrix.max(axis=1)

    plagiarism_score = float(np.mean(max_sim) * 100)

    return {
        "plagiarism_percentage": round(plagiarism_score, 2)
    }

# =======================
# DOWNLOAD REPORT
# =======================

@app.get("/download-report")
def download_report():
    return FileResponse(
        os.path.join(UPLOAD_DIR, "report.html"),
        media_type="text/html",
        filename="TrueText_Report.html"
    )

@app.post("/admin/login")
async def admin_login(request: Request, username: str = Form(...), password: str = Form(...)):

    conn = sqlite3.connect("truetext.db", check_same_thread=False)
    cursor = conn.cursor()

    # get admin from DB
    cursor.execute(
        "SELECT password FROM users WHERE username=? AND role='admin'",
        (username,)
    )

    admin = cursor.fetchone()
    conn.close()

    # verify password
    if admin and bcrypt.checkpw(password.encode(), admin[0]):
        request.session["admin"] = True
        return {"message": "Admin login successful"}

    raise HTTPException(status_code=401, detail="Invalid admin credentials")

@app.get("/admin/logout")
async def logout(request: Request):
    request.session.clear()
    return {"message": "Logged out"}

# =======================
# USER SIGNUP
# =======================


@app.post("/user/signup")
async def user_signup(username: str = Form(...), password: str = Form(...)):

    conn = sqlite3.connect("truetext.db", check_same_thread=False)
    cursor = conn.cursor()

    # HASH PASSWORD HERE
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    try:
        cursor.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, hashed_password, "user")
        )
        conn.commit()
        conn.close()

        return {"status": "success", "message": "User account created"}

    except sqlite3.IntegrityError:
        conn.close()
        return {"status": "error", "message": "Username already exists"}
    
# =======================
# USER LOGIN
# =======================

@app.post("/user/login")
async def user_login(request: Request, username: str = Form(...), password: str = Form(...)):

    conn = sqlite3.connect("truetext.db", check_same_thread=False)
    cursor = conn.cursor()

    # ONLY fetch hashed password
    cursor.execute(
        "SELECT password FROM users WHERE username=?",
        (username,)
    )

    user = cursor.fetchone()
    conn.close()

    # VERIFY HASH
    if user and bcrypt.checkpw(password.encode(), user[0]):
        request.session["user"] = username
        return {"status": "success", "message": "User login successful"}

    raise HTTPException(status_code=401, detail="Invalid username or password")
    
# =======================
# USER LOGOUT
# =======================

@app.get("/user/logout")
async def user_logout(request: Request):
    request.session.pop("user", None)
    return {"message": "User logged out"}