FROM python:3.7-slim-buster
# Install python packages
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
EXPOSE 5000
RUN python -m pip install psycopg2-binary  mlflow==1.11.0

