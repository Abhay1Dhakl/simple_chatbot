FROM python:3.10

#Set the working directory in the container
WORKDIR /app

#Copy the current directory contents into the container at /app
COPY . /app


RUN pip install -r requirement.txt


#Command to run application
CMD ["python", "app.py"]