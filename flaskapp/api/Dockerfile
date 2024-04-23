FROM python:3.9-slim

# Set an argument for the source directory and env variable for the source
ARG SOURCE_DIR= .
ENV SOURCE_DIR=${SOURCE_DIR}

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=development

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY ${SOURCE_DIR}/requirements.txt .


# Install any dependencies
# RUN pip install tensorflow tensorflow.io
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY ${SOURCE_DIR} .

# Expose port 5000 to the outside world
EXPOSE 5000

# Run the Flask application
CMD ["flask", "run"]
# CMD [ "python", "app.py" ]
# CMD ["sh", "-c", "ls && pwd"]
# CMD ["ls"]