# Dockerfile
# filepath: /home/electricdystopia/AI_Tutor/AI_Tutor/Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# copy app sources
COPY . .
ENV PORT=8080
EXPOSE 8080
# run streamlit on port 8080 bound to 0.0.0.0
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]