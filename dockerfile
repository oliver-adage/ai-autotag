FROM databricksruntime/python:latest
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV DBX_PY=/databricks/python3/bin/python
ENV DBX_PIP=/databricks/python3/bin/pip

# upgrade pip/setuptools/wheel to get the best available wheels
RUN $DBX_PIP install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .
RUN $DBX_PIP install --no-cache-dir -r requirements.txt

# model + labels
RUN mkdir -p models && \
    curl -L -o models/mobilenetv2-7.onnx https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx && \
    curl -L -o models/imagenet-simple-labels.json https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json

COPY app.py .
ENV APP_PORT=8000
EXPOSE 8000
CMD ["/databricks/python3/bin/python","-c","print('Container ready; start app via init script.')"]
