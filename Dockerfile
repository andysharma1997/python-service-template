FROM tensorflow/tensorflow:latest
MAINTAINER venkataramana@salesken.ai
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8002
CMD ["gunicorn","--bind","0.0.0.0:8002","--workers=2","main:app"]