FROM python:3.11

WORKDIR /code

COPY . /code

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "src.main:app", "--host", "127.0.0.1", "--port", "80"]