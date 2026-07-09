@echo off
REM Reddit Signal Agent - local dashboard server (http://127.0.0.1:8765)
cd /d "%~dp0"
"C:\Users\userAdmin\AppData\Local\Programs\Python\Python310\python.exe" -m reddit_agent.server
