FROM python:3.8

WORKDIR /app

COPY . .

RUN apt-get update -y
RUN apt-get upgrade -y

RUN apt install cmake -y
RUN apt install libgl1-mesa-glx -y

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python","server.py"]
