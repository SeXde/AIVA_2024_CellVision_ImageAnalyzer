FROM python:3.11

EXPOSE 80
WORKDIR /code

COPY . /code

RUN apt update; apt install -y libgl1
RUN pip3 install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install python-multipart

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]