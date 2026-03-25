from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from predict import DEFAULT_MODEL_PATH, predict_image
import cv2
import numpy as np
try:
    from ultralytics import YOLO
    YOLO_MODEL = YOLO("yolov8n.pt")
except ImportError:
    YOLO_MODEL = None

ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(ROOT / "templates"))
app = FastAPI(title="Drone vs Bird Classifier")
DASHBOARD_STATE = {
    "scans": 0,
    "birds": 0,
    "drones": 0,
    "accuracy_sum": 0.0,
    "recent": [],
}
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def load_dashboard_stats() -> dict:
    return {
        "stats": {
            "scans": DASHBOARD_STATE["scans"],
            "birds": DASHBOARD_STATE["birds"],
            "drones": DASHBOARD_STATE["drones"],
            "avg_accuracy": (
                DASHBOARD_STATE["accuracy_sum"] / DASHBOARD_STATE["scans"] * 100
                if DASHBOARD_STATE["scans"]
                else 0.0
            ),
        },
        "recent_detections": DASHBOARD_STATE["recent"][:5],
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    context = load_dashboard_stats()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
            "error": None,
            "model_ready": DEFAULT_MODEL_PATH.exists(),
            "status_message": "Ready to detect." if DEFAULT_MODEL_PATH.exists() else None,
            "uploaded_image": None,
            **context,
        },
    )


@app.post("/", response_class=HTMLResponse)
async def classify(request: Request, image: UploadFile = File(...)) -> HTMLResponse:
    context = load_dashboard_stats()
    if not DEFAULT_MODEL_PATH.exists():
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": None,
                "error": f"Model not found at {DEFAULT_MODEL_PATH}. Run train.py first.",
                "model_ready": False,
                "status_message": None,
                "uploaded_image": None,
                **context,
            },
            status_code=400,
        )

    try:
        suffix = Path(image.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "result": None,
                    "error": None,
                    "model_ready": True,
                    "status_message": "Invalid file type.",
                    "uploaded_image": None,
                    **context,
                },
                status_code=400,
            )

        raw_bytes = await image.read()
        uploaded = Image.open(BytesIO(raw_bytes)).convert("RGB")
        result = predict_image(uploaded)
        DASHBOARD_STATE["scans"] += 1
        DASHBOARD_STATE["accuracy_sum"] += float(result["confidence"])
        if result["label"] == "bird":
            DASHBOARD_STATE["birds"] += 1
        elif result["label"] == "drone":
            DASHBOARD_STATE["drones"] += 1
        DASHBOARD_STATE["recent"].insert(
            0,
            {
                "label": result["label"],
                "confidence": float(result["confidence"]),
                "file_name": image.filename or "uploaded image",
            },
        )
        DASHBOARD_STATE["recent"] = DASHBOARD_STATE["recent"][:8]
        context = load_dashboard_stats()
        encoded_image = base64.b64encode(raw_bytes).decode("ascii")
        uploaded_image = f"data:image/{suffix.lstrip('.')};base64,{encoded_image}"
    except Exception as exc:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": None,
                "error": f"Could not process the image: {exc}",
                "model_ready": True,
                "status_message": "Invalid file type.",
                "uploaded_image": None,
                **context,
            },
            status_code=400,
        )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "error": None,
            "model_ready": True,
            "status_message": f"Detected - it's a {result['label']}.",
            "uploaded_image": uploaded_image,
            **context,
        },
    )

@app.post("/api/detect")
async def api_detect(image: UploadFile = File(...)):
    if not DEFAULT_MODEL_PATH.exists():
        return JSONResponse({"error": f"Model not found at {DEFAULT_MODEL_PATH}. Run train.py first."}, status_code=400)

    try:
        suffix = Path(image.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return JSONResponse({"error": "Invalid file type."}, status_code=400)

        raw_bytes = await image.read()
        uploaded = Image.open(BytesIO(raw_bytes)).convert("RGB")
        
        has_box = False
        cv_img = cv2.cvtColor(np.array(uploaded), cv2.COLOR_RGB2BGR)
        
        if YOLO_MODEL is not None:
            results = YOLO_MODEL(uploaded, verbose=False)
            boxes = results[0].boxes
            if len(boxes) > 0:
                confidences = boxes.conf.cpu().numpy()
                best_idx = int(np.argmax(confidences))
                box = boxes.xyxy[best_idx].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                
                cropped = uploaded.crop((x1, y1, x2, y2))
                result = predict_image(cropped)
                
                color = (0, 0, 255) if result["label"] == "drone" else (0, 255, 0)
                thickness = max(2, int(min(cv_img.shape[0], cv_img.shape[1]) * 0.005))
                cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, thickness)
                
                label_text = f"{result['label'].upper()} ({result['confidence']*100:.1f}%)"
                font_scale = max(0.5, min(cv_img.shape[0], cv_img.shape[1]) * 0.001)
                cv2.putText(cv_img, label_text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, max(1, int(thickness/2)))
                
                uploaded = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
                has_box = True
                
        if not has_box:
            result = predict_image(uploaded)

        is_live = (image.filename == "live_snapshot.jpg")

        if not is_live:
            DASHBOARD_STATE["scans"] += 1
            DASHBOARD_STATE["accuracy_sum"] += float(result["confidence"])
            if result["label"] == "bird":
                DASHBOARD_STATE["birds"] += 1
            elif result["label"] == "drone":
                DASHBOARD_STATE["drones"] += 1
                
            DASHBOARD_STATE["recent"].insert(
                0,
                {
                    "label": result["label"],
                    "confidence": float(result["confidence"]),
                    "file_name": image.filename or "uploaded image",
                },
            )
            DASHBOARD_STATE["recent"] = DASHBOARD_STATE["recent"][:8]
        
        img_byte_arr = BytesIO()
        uploaded.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        encoded_image = base64.b64encode(img_byte_arr).decode("ascii")
        uploaded_image_data = f"data:image/jpeg;base64,{encoded_image}"
        
        return {
            "result": result,
            "uploaded_image": uploaded_image_data,
            "stats": load_dashboard_stats()
        }
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"Could not process the image: {exc}"}, status_code=400)

