# 📐 Math PDF Question Extractor

An AI-powered full-stack web application that extracts mathematics questions from a PDF exam paper and presents them as **cropped images** or **structured text with LaTeX rendering**.

---

## 🖥️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 19 + TypeScript, Vite, KaTeX |
| **Backend** | Python FastAPI, Uvicorn |
| **AI Engine** | Google Gemini 2.5 Flash (`google-genai`) |
| **PDF Engine** | PyMuPDF (fitz), Pillow |
| **Design** | Classic Chalkboard Dark Theme — Lora · Montserrat · Fira Code |

---

## 🚀 Quick Start

### 1. Clone / Open the project

Set the active workspace to:
```
C:\Users\HP\.gemini\antigravity\scratch\math-pdf-extractor
```

### 2. Configure the Backend

**Edit `backend/.env`** and fill in your Gemini API key:
```env
GEMINI_API_KEY=your_actual_api_key_here
```
Get a key at → https://ai.google.dev

### 3. Start the Backend

Open a terminal and run:
```powershell
cd backend
py -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### 4. Start the Frontend

Open a **second terminal** and run:
```powershell
cd frontend
npm run dev
```

The app will be available at `http://localhost:5173`

### 5. Generate a Test PDF (optional)

A sample math exam PDF is included for testing:
```powershell
py generate_test_pdf.py
# → Creates test_math.pdf in the project root
```

---

## 📋 Features

### Form Inputs
- **Name** — identifies the request
- **Number of Questions** — how many to extract (1–50)
- **Output Format** — choose between:
  - 🖼️ **Cropped Image** — each question block is rendered as a high-res PNG
  - 📝 **Structured Text** — equations rendered via KaTeX, diagrams shown alongside text
- **PDF Upload** — drag-and-drop or click to browse (PDF only)
- **Gemini API Key** — supply directly in-browser if not set in `.env`

### AI Pipeline (per PDF page)
1. PyMuPDF renders the page at 2× zoom to a high-res PNG
2. The page image is sent to **Gemini 2.5 Flash** with a structured JSON schema
3. Gemini returns bounding boxes (`[ymin, xmin, ymax, xmax]` in 0–1000 space) for each question and any diagrams
4. Pillow crops the question block and diagram separately
5. Crops are served via FastAPI's `/static/crops/` endpoint

### LaTeX Rendering
- Inline: `$...$`
- Block/display: `$$...$$`
- Powered by **KaTeX** loaded from CDN (no npm dependency needed)

---

## 📁 Project Structure

```
math-pdf-extractor/
├── backend/
│   ├── app.py              ← FastAPI application
│   ├── requirements.txt    ← Python dependencies
│   ├── .env                ← Gemini API key (not committed)
│   ├── uploads/            ← Temp PDF storage (auto-created)
│   └── static/crops/       ← Served cropped images (auto-created)
├── frontend/
│   ├── src/
│   │   ├── App.tsx         ← Main application UI
│   │   ├── MathRenderer.tsx← KaTeX LaTeX renderer component
│   │   ├── index.css       ← Classic Chalkboard design system
│   │   └── main.tsx        ← React entrypoint
│   ├── index.html          ← SEO-optimised HTML shell
│   └── package.json
├── generate_test_pdf.py    ← Creates a sample math exam PDF
└── README.md
```

---

## 🎨 Design Palette

| Token | Value | Use |
|-------|-------|-----|
| `--bg-page` | `#1A1E24` | Page background |
| `--bg-surface` | `#22272F` | Card surfaces |
| `--teal` | `#8FBCBB` | Primary accent, active states |
| `--gold` | `#EBCB8B` | Submit button, step numbers |
| `--text-primary` | `#E5E9F0` | Body text |
| `--font-body` | Lora (serif) | Questions, prose |
| `--font-ui` | Montserrat | Headings, labels, buttons |
| `--font-mono` | Fira Code | LaTeX code, filenames |
