@echo off
wsl -d Ubuntu -e bash -c "cd /home/pawl/club-3090/services/litellm && docker compose down && docker compose up -d --force-recreate"
pause
