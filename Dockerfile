FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim
# FROM python:3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv


WORKDIR /app

COPY ["Pipfile", "Pipfile.lock" , "./"]

RUN pipenv install --system --deploy

COPY ["starter.py", "./"]

EXPOSE 9696 

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "starter:app"]