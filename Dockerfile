FROM python:3.11.1-slim

WORKDIR /usr/src/flask

ARG PORT
ENV PORT=$PORT

COPY ./requirements.txt /usr/src/flask/requirements.txt
RUN pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /usr/src/flask/requirements.txt

COPY ./app /usr/src/flask/app

ENV PYTHONPATH "${PYTHONPATH}:/usr/src/flask/app"

CMD uvicorn main:app --reload --host 0.0.0.0 --port $PORT