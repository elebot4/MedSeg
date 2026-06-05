FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /app

# Install package in editable mode with ONNX extras.
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY scripts ./scripts
RUN python -m pip install --upgrade pip && pip install -e ".[onnx]"

# copy remaining source code
COPY . .

# add default nonroot user w/ uid & pid 1000
RUN groupadd --gid 1000 nonroot \
    && useradd --uid 1000 --gid 1000 -m nonroot
RUN apt-get update \
    && apt-get install -y sudo \
    && echo nonroot ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/nonroot \
    && chmod 0440 /etc/sudoers.d/nonroot

# switch to created nonroot user
USER nonroot
EXPOSE 8080

# run API server by default
CMD ["python", "src/serve.py", "--backend", "pytorch", "--device", "cpu", "--host", "0.0.0.0", "--port", "8080"]
