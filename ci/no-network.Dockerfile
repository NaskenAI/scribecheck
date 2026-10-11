# Test image for the no-network CI job. Dependencies are installed while building
# the image; the tests then run with `docker run --network none`.
FROM python:3.12-slim
WORKDIR /src
COPY . .
RUN python -m pip install --no-cache-dir -e ".[dev]"
CMD ["python", "-m", "pytest", "-q"]
