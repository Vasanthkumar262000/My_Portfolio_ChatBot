# How to Run Your Portfolio ChatBot Project

## Step-by-Step Guide

### Step 1: Start Backend Server

Open PowerShell/CMD and navigate to project folder:
```powershell
cd E:\project\My_Portfolio_ChatBot\My_Portfolio_ChatBot
```

Start the backend:
```powershell
.\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

**Keep this terminal open!** You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

### Step 2: Start Cloudflare Tunnel

Open a **NEW** PowerShell/CMD window and run:
```bash
cloudflared tunnel run My-Portfolio-ChatBot
```

**Keep this terminal open too!** You should see connection messages.

---

### Step 3: Verify It's Working

Open your browser and test:
```
https://chatbot.vasanthkumarr.com/health
```

You should see JSON response with status information.

---

### Step 4: Use Your Frontend

Your frontend should connect to:
```
https://chatbot.vasanthkumarr.com
```

---

## Important Notes

- **Both terminals must stay running** - Backend in Terminal 1, Tunnel in Terminal 2
- **To stop:** Press `Ctrl+C` in both terminals
- **If backend stops:** Restart Step 1
- **If tunnel stops:** Restart Step 2

---

## Adding New Documents

To add new PDF or DOCX files to the vector database:

1. **Stop backend** (Ctrl+C in Terminal 1)

2. **Add document:**
   ```powershell
   .\venv\Scripts\python.exe add_document.py data\your_file.pdf
   ```
   or
   ```powershell
   .\venv\Scripts\python.exe add_document.py data\your_file.docx
   ```

3. **Restart backend** (Step 1 again)

---

## That's It!

Your chatbot is now running and accessible via your custom domain.

