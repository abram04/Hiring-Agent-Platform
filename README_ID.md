# Hiring Agent Platform

AI agent berbasis kecerdasan buatan yang secara otomatis mencocokkan kandidat dengan lowongan kerja yang relevan menggunakan vector similarity search dan LLM re-ranking.

Dibangun sebagai portfolio project untuk mendemonstrasikan kemampuan AI agent engineering di level produksi.

---

## Tech Stack

| Teknologi | Fungsi |
|-----------|--------|
| **LangGraph** | Orkestrasi workflow agent |
| **PostgreSQL + pgvector** | Vector database untuk similarity search |
| **Google Gemini** | Generate embedding & LLM re-ranking |
| **FastAPI** | REST API server |
| **LangSmith** | Monitoring & tracing agent |
| **Docker** | Containerisasi database |
| **asyncpg** | Async PostgreSQL driver |

---

## Cara Kerja

```
Resume kandidat (teks atau PDF)
        ↓
Node 1: Generate embedding 768 dimensi via Gemini
        ↓
Node 2: Vector cosine similarity search di pgvector
        ↓
Node 3: LLM re-ranking + reasoning via Gemini
        ↓
JSON response berisi top matches + penjelasan
```

### Kenapa Dua Tahap Ranking?

Vector similarity murni mencari job dengan teks yang mirip — tapi tidak bisa reasoning tentang konteks. Misalnya, posisi "Full Stack Developer" mungkin mendapat skor tinggi untuk kandidat backend karena kesamaan keyword, tapi LLM memahami bahwa kandidat tidak memiliki skill frontend.

Dengan menggabungkan vector search (retrieval yang cepat dan scalable) dengan LLM re-ranking (reasoning kontekstual), agent menghasilkan hasil yang lebih akurat dan dapat dijelaskan.

---

## Struktur Project

```
hiring_agent/
├── main.py              ← FastAPI server + LangGraph agent
├── agent.py             ← Agent standalone (untuk testing)
├── test_db.py           ← Test koneksi database
├── test_embedding.py    ← Test generate embedding
├── test_search.py       ← Test vector search
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup & Instalasi

### Prasyarat
- Python 3.11+
- Docker Desktop
- Google AI API Key (dari https://aistudio.google.com/apikey)
- LangSmith API Key (dari https://smith.langchain.com) — opsional, untuk monitoring

### 1. Clone repository
```bash
git clone https://github.com/yourusername/hiring-agent.git
cd hiring-agent
```

### 2. Buat virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Konfigurasi environment variables
```bash
cp .env.example .env
```

Edit file `.env` dan isi credentials kamu:
```
DATABASE_URL=postgresql://hiring_user:hiring_pass@localhost:5432/hiring_agent
GOOGLE_API_KEY=api_key_google_kamu
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=api_key_langsmith_kamu
LANGCHAIN_PROJECT=hiring-agent
```

### 5. Jalankan PostgreSQL + pgvector dengan Docker
```bash
docker run -d --name hiring_agent_db \
  -e POSTGRES_USER=hiring_user \
  -e POSTGRES_PASSWORD=hiring_pass \
  -e POSTGRES_DB=hiring_agent \
  -p 5432:5432 \
  pgvector/pgvector:0.8.2-pg18-trixie
```

### 6. Inisialisasi database
Jalankan SQL schema di DBeaver atau psql:
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE job_listings (
    job_id      SERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    company     TEXT NOT NULL,
    location    TEXT,
    job_type    TEXT,
    description TEXT NOT NULL,
    requirements TEXT[],
    embedding   vector(768)
);

CREATE TABLE candidates (
    candidate_id    SERIAL PRIMARY KEY,
    name            TEXT,
    email           TEXT UNIQUE,
    resume_text     TEXT,
    parsed_profile  JSONB,
    embedding       vector(768),
    status          TEXT DEFAULT 'new'
);
```

### 7. Jalankan server
```bash
python main.py
```

### 8. Buka dokumentasi API
```
http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/` | Health check |
| POST | `/match-jobs` | Match jobs dari plain text resume |
| POST | `/match-jobs-pdf` | Match jobs dari upload file PDF |

### Contoh Request — Plain Text
```bash
curl -X POST http://localhost:8000/match-jobs \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_name": "Budi Santoso",
    "resume_text": "Python developer dengan 5 tahun pengalaman di FastAPI dan PostgreSQL."
  }'
```

### Contoh Request — Upload PDF
```bash
curl -X POST "http://localhost:8000/match-jobs-pdf?candidate_name=Budi%20Santoso" \
  -F "file=@cv_budi.pdf"
```

### Contoh Response
```json
{
  "candidate_name": "Budi Santoso",
  "matches": [
    {
      "title": "Senior Python Engineer",
      "company": "TechCorp Indonesia",
      "location": "Jakarta",
      "job_type": "full-time",
      "similarity": 0.7314
    }
  ],
  "llm_analysis": "RANKING:\n1. Senior Python Engineer — Sangat cocok karena keahlian Python dan FastAPI kandidat...\n\nREKOMENDASI:\nPrioritaskan posisi Senior Python Engineer."
}
```

---

## Arsitektur Agent

```
FastAPI Gateway
      │
      ▼
LangGraph Agent (AgentState)
      │
      ├── Node 1: embed_resume
      │     └── Gemini text-embedding-001 (768 dimensi)
      │
      ├── Node 2: search_jobs
      │     └── pgvector cosine similarity (operator <=>)
      │           └── ivfflat index (lists=10)
      │
      └── Node 3: llm_rerank
            └── Gemini LLM
                  ├── Re-ranking kontekstual
                  ├── Reasoning per match
                  └── Rekomendasi final
```

---

## Monitoring

Project ini terintegrasi dengan **LangSmith** untuk observabilitas agent secara penuh:
- Trace setiap eksekusi node
- Monitor latency per node
- Track penggunaan token dan biaya API
- Debug kegagalan agent

Akses traces kamu di: https://smith.langchain.com

---

## Skill yang Didemonstrasikan

- ✅ LangGraph agent workflow dengan typed state
- ✅ Desain dan indexing vector database (pgvector + ivfflat)
- ✅ Generate embedding dan semantic search
- ✅ Integrasi LLM dengan structured prompting
- ✅ Async Python patterns (asyncpg, FastAPI)
- ✅ Desain REST API dengan dokumentasi otomatis
- ✅ Parsing PDF dan ekstraksi teks
- ✅ Production monitoring dengan LangSmith
- ✅ Docker containerization

---

## Lisensi

MIT License — silakan gunakan project ini sebagai referensi untuk portfolio AI agent kamu sendiri.
