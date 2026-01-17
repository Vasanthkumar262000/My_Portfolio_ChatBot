@echo off
echo Starting Cloudflare Tunnel...
echo.
echo Make sure your backend is running on port 8000 first!
echo.
cloudflared tunnel --url http://localhost:8000
pause






