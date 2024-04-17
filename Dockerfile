# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR "/src"

# Copy the requirements file into the container at /src
COPY requirements.txt /src/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /src
COPY . /src/

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME cellvision-dev

# Set the maintainer label
LABEL maintainer="avberdote <avberdote@hotmail.com>"

# Run tests
# RUN python -m unittest discover tests/

# Run main.py when the container launches
CMD ["pwd"]
CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "80"]
