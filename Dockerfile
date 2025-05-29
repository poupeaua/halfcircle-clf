FROM python:3.11-slim

ARG APP_DIR=/app

WORKDIR ${APP_DIR}

COPY ./requirements.txt ${APP_DIR}/requirements.txt

RUN pip install --no-cache-dir --upgrade -r ${APP_DIR}/requirements.txt

COPY ./models ${APP_DIR}/models
COPY ./src ${APP_DIR}/src
COPY ./app.py ${APP_DIR}/app.py

CMD ["fastapi", "run", "${APP_DIR}/app.py", "--port", "80"]

