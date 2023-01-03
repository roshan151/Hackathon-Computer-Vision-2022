# To build image:
# $ docker build -f Dockerfile . -t price_app

# A. To start and run in background
# $ docker run -d -p 8080:8080 price_app
# Issue request to http://0.0.0.0:8080/predict

# Here image_port = 8080, image_name = price_app


# B. To start without starting the app
# Use if you want to be able to watch progress from CLI
# $ docker run -d -p 8080:8080 price_app /bin/bash
# $ docker exec -it <container_id> /bin/bash
# $ (base) root@<some numbers>:/# python app.py run -h 0.0.0.0 -p 8080 --no-debugger
# Issue request to http://0.0.0.0:<image_port/predict

# Import base image
FROM python:3.8.7-slim-buster as base

# Set working directory
WORKDIR /usr/src/app

# Set environment variables to reduce size
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set up environment
COPY ./requirements.txt /usr/src/app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

FROM base

# Copy app and model files
COPY ./service.py /usr/src/app/app.py

# TODO: add model files to docker container
COPY ./model/ /usr/src/app/.

CMD [ "python3", "-m", "flask", "run", "--host=0.0.0.0", "--port=8080"]
