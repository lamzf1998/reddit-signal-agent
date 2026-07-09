@echo off
REM Reddit Signal Agent - hourly digest launcher (for Windows Task Scheduler).
REM Runs the bot from the project root and appends output to a log file.

cd /d "%~dp0"
if not exist "reddit_agent\.state" mkdir "reddit_agent\.state"

echo ---- run started %DATE% %TIME% ---->> "reddit_agent\.state\run.log"
"C:\Users\userAdmin\AppData\Local\Programs\Python\Python310\python.exe" -m reddit_agent.main >> "reddit_agent\.state\run.log" 2>&1

REM Publish the snapshot to GitHub Pages (only if this folder is a git repo).
if exist "%~dp0.git" (
  git -C "%~dp0" add docs/signals.json >> "reddit_agent\.state\run.log" 2>&1
  git -C "%~dp0" commit -m "update signals snapshot" >> "reddit_agent\.state\run.log" 2>&1
  git -C "%~dp0" push >> "reddit_agent\.state\run.log" 2>&1
)

echo ---- run finished %DATE% %TIME% ---->> "reddit_agent\.state\run.log"
