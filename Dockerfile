# Use an official PyTorch base image
FROM python:3.14
RUN pip install poetry
RUN mkdir -p /app
COPY pyproject.toml /app
COPY poetry.lock /app
COPY poetry.toml /app
WORKDIR /app
RUN poetry install
# Copy the local files to the Docker container
COPY ./examples /app/examples
COPY ./ltn_imp /app/ltn_imp
# Set an environment variable for the output directory
ENV OUTPUT_DIR=/output
# Create the output directory
RUN mkdir -p $OUTPUT_DIR
# Declare a volume for the output directory
VOLUME ["/output"]
ENV EXPERIMENT_NAME="default"
# Command to run the Python script
CMD poetry run python -m examples.${EXPERIMENT_NAME}.training_script
