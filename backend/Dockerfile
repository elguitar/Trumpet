FROM tensorflow/tensorflow:2.3.1

WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8081

CMD ["uvicorn", "api:app","--reload", "--port", "8081", "--host", "0.0.0.0"]
