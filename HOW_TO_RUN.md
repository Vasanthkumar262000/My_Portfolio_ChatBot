# How to Run This Project & Connect to a Frontend

## Local vs Cloud: What You Need to Know

| | **Local backend** (your machine) | **Cloud Run** (Google’s servers) |
|---|-----------------------------------|----------------------------------|
| **Restarting your backend** | Only affects your local server. | Does **not** update Cloud Run. Cloud Run runs a separate copy of your app. |
| **Updating the cloud** | N/A | Run a **redeploy** (e.g. `./deploy.sh` or `gcloud run deploy ...`). That builds a new image and updates the live service. |
| **Keeping the chatbot alive** | Yes — if you stop the process or shut your computer, the chatbot is down. | **No.** Once deployed, Google keeps the service available. You do **not** need your laptop or local backend running. The chatbot stays up 24/7 (or scales to zero and wakes on request). |
| **After reindexing documents** | Restart the local server to load the new index. | **Redeploy** so the new `data/faiss_index` is baked into the container and Cloud Run serves the updated index. |

**Summary:**  
- **Local:** Restart = only local. Your machine must be running for the chatbot to work.  
- **Cloud:** Redeploy = update cloud. You do **not** need your backend running; Cloud Run keeps the chatbot alive.

---

## 1. Run the Backend Locally

### Prerequisites
- Python 3.9+
- A **Groq API key** (or Vertex AI) for the LLM – get one at [console.groq.com](https://console.groq.com)

### Steps

**1. Create and activate a virtual environment**

```bash
cd /path/to/My_Portfolio_ChatBot
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Configure environment**

Create a `.env` file in the project root (or copy from an example):

```env
# Required for chat – at least one of these
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant

# Optional: Google Vertex AI (backup LLM)
# VERTEX_AI_PROJECT=your-gcp-project-id
# GEMINI_MODEL=gemini-1.5-flash

# Optional: local dev
ENVIRONMENT=development
PORT=8080
```

**3. Load documents (required for chat to use your content)**

The vector store starts empty until you add a document. Use either script:

- **Add a document** (PDF or DOCX; recommended – uses improved chunking and hybrid search):
  ```bash
  python add_document.py "data/Vasanth information doc.pdf"
  ```
- **Re-train from scratch** (clear index and re-add one document; use if chat answers were wrong or content wasn’t found):
  ```bash
  python add_document.py "data/Vasanth information doc.pdf" --reindex
  ```

Alternatively, you can still use:
```bash
python process_pdf.py "data/Vasanth information doc.pdf"
```

Without this step, the backend will show "Documents: 0" and chat won’t have your portfolio content.

**If the chatbot always says "I don't have specific information about that":**  
The training document may have PDF extraction errors (e.g. "Loca4on", "sokware"). The pipeline now corrects these. **Reindex** so the vector store uses corrected text:

```bash
# Prefer DOCX if you have it (usually cleaner than PDF)
python add_document.py "data/Vasanth information doc.docx" --reindex
# Or with PDF (artifacts are auto-corrected)
python add_document.py "data/Vasanth information doc.pdf" --reindex
```

Then restart the backend (and redeploy to Cloud Run if that’s where you run the chatbot).

**4. Start the server**

```bash
# Default port 8080 (or set PORT in .env)
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

Or run the app directly (uses `PORT` from env, default 8080):

```bash
python main.py
```

- API: **http://localhost:8080**
- Interactive docs: **http://localhost:8080/docs**
- ReDoc: **http://localhost:8080/redoc**

---

## 2. Connect Your Frontend to This Backend

### Base URL

- **Local:** `http://localhost:8080`
- **Cloud Run:** your Cloud Run service URL (e.g. `https://chatbot-api-xxxxx-uc.a.run.app`)

### Using your Google Cloud Run URL

1. **Set the URL in your frontend project** (so you can switch local vs production):

   **Next.js** – in `.env.local` or in your hosting provider’s env vars:
   ```env
   NEXT_PUBLIC_API_URL=https://YOUR-SERVICE-xxxxx-uc.a.run.app
   ```

   **Vite / Create React App** – in `.env` or `.env.production`:
   ```env
   VITE_API_URL=https://YOUR-SERVICE-xxxxx-uc.a.run.app
   # or REACT_APP_API_URL=... for CRA
   ```

2. **Use it in code** (one base URL for all API calls):

   **Next.js:**
   ```ts
   const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
   ```

   **Vite / React:**
   ```ts
   const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8080';
   ```

### CORS

The backend already allows cross-origin requests:

- In **development** (`ENVIRONMENT=development`), it allows all origins (`*`).
- In **production**, it still has `allow_origins=["*"]` in `main.py` (line 125). For production you can change this in `main.py` to your real frontend domains (e.g. `https://vasanthkumarr.com`).

So a frontend on `http://localhost:3000` or your deployed site can call this API without extra CORS config on the backend, as long as you don’t restrict origins to a specific list.

### Main endpoints for the frontend

| Method | Endpoint      | Purpose |
|--------|---------------|--------|
| `GET`  | `/`           | API info and status |
| `GET`  | `/health`     | Health check |
| `POST` | `/chat`       | Send a message, get RAG reply |
| `POST` | `/query`      | Semantic search (no LLM) |
| `GET`  | `/stats`      | Service stats |
| `POST` | `/upload`     | Upload a PDF (optional) |

### Example: Chat from the frontend

**Request**

```http
POST /chat
Content-Type: application/json

{ "message": "What is your experience?" }
```

**Response**

```json
{
  "response": "...",
  "sources": [...],
  "model": "llama-3.1-8b-instant",
  "provider": "groq"
}
```

**JavaScript/TypeScript example**

```javascript
const response = await fetch(`${API_BASE}/chat`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: userMessage }),
});
const data = await response.json();
console.log(data.response, data.sources);
```

### Optional: Expose local backend to the internet (e.g. for a deployed frontend)

If your frontend is deployed but you’re running the backend only on your machine, you can expose it with a tunnel:

- **Cloudflare Tunnel** (project has `start_tunnel.bat` for Windows):
  - Start backend: `uvicorn main:app --host 0.0.0.0 --port 8000`
  - Run: `cloudflared tunnel --url http://localhost:8000`
  - Use the generated `https://xxx.trycloudflare.com` URL as `API_BASE` in your frontend.

- **ngrok:** `ngrok http 8080` and use the `https://xxx.ngrok.io` URL as `API_BASE`.

For production, deploy the backend (e.g. to Cloud Run) and set your frontend’s `API_BASE` to that Cloud Run URL.

---

## 3. Quick checklist

- [ ] Python venv created and `pip install -r requirements.txt` done
- [ ] `.env` with `GROQ_API_KEY` (or Vertex AI) set
- [ ] Optional: `python process_pdf.py data/your_document.pdf`
- [ ] Backend running: `uvicorn main:app --reload --host 0.0.0.0 --port 8080`
- [ ] Frontend uses `API_BASE = http://localhost:8080` (or your Cloud Run URL)
- [ ] Chat: `POST /chat` with `{ "message": "..." }`

For deployment to Google Cloud Run, see **CLOUD_RUN_DEPLOY.md**.
