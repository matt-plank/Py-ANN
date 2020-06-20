FROM ubuntu:latest

# Copy Repository Over
COPY . .

# Install Dependencies
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN bash ./install.sh

# Entrypoint
CMD ["python3", "./examples/xor.py"]