FROM cgr.dev/chainguard-private/python:latest-dev AS builder

USER root
RUN mkdir -p /deps

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/deps -r requirements.txt

FROM cgr.dev/chainguard-private/python:latest-dev

USER root
WORKDIR /app

COPY --from=builder /deps/lib /usr/local/lib
COPY --from=builder /deps/bin /usr/local/bin

COPY main.py .
COPY demo_data.json .

USER nonroot
EXPOSE 8080

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
