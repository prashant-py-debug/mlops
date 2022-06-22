FROM python:3.9.7-slim

WORKDIR /app

COPY ["requirements.txt","./"]

RUN pip install -r requirements.txt

COPY ["model.bin","parameterised.py","./"]

CMD ["python", "parameterised.py", "2021","04"]