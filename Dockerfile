FROM python:3.8-slim
WORKDIR /

RUN apt-get update;

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt;
COPY . .
# set PYTHONPATH
ENV PYTHONPATH /
ENV FLASK_APP retrainer/app/app.py
ENV PYTHONUNBUFFERED true

# expose Flask App
EXPOSE 5004


# Run the application
CMD ["gunicorn", "-t", "0", "-b" , "0.0.0.0:5004", "-R", "--log-level", "debug", "retrainer.app.app:app"]
