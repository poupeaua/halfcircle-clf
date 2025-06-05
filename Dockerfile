FROM python:3.13-alpine

ARG APP_DIR=/app

WORKDIR ${APP_DIR}

RUN pip install uv

COPY ./requirements-api.txt ${APP_DIR}/requirements.txt

RUN uv pip install -r requirements.txt --system

COPY ./models ${APP_DIR}/models
COPY ./src ${APP_DIR}/src
COPY ./api.py ${APP_DIR}/api.py

EXPOSE 80
CMD ["fastapi", "run", "/app/api.py", "--port", "80"]
