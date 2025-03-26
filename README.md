# NTPD_LAB04

# Uruchomienie lokalnie
uvicorn app:app --host 0.0.0.0 --port 8000

# Uruchomienie przez Dockera
docker build -t my-ml-app .
docker run -d -p 8000:8000 my-ml-app

# Uruchomienie przez Docker Compose
docker-compose up -d
