FROM tensorflow/tensorflow:latest
MAINTAINER venkataramana@salesken.ai
RUN pip install -r requirements.txt
COPY ./app /app
EXPOSE 8002
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]