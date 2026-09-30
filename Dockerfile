FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -c "from rfdetr import RFDETRBase; RFDETRBase()" || true
RUN python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')" || true
EXPOSE 8000 8501
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0"]
