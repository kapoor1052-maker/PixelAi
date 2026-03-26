from dotenv import load_dotenv
load_dotenv()
import os
import uuid
import time
import base64
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from batch_utils import (
    validate_image,
    compute_metrics,
    create_zip_batch,
    create_batch_executor,
)

# =========================
# OPTIONAL POSTGRESQL
# =========================
import psycopg2



psycopg2 = None
RealDictCursor = None
PSYCOPG2_AVAILABLE = False

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    pass

# =========================
# APP
# =========================
app = FastAPI(title="AI Image Enhancer")

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
STATIC_DIR = Path("static")

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

BATCH_EXECUTOR = create_batch_executor(max_workers=2)
CLEANUP_INTERVAL = 3600 * 24  # 24 hours

# =========================
# DATABASE CONFIG
# =========================
DEFAULT_LOCAL_DB_URL = "postgresql://postgres:YOUR_PASSWORD@localhost:5432/image_enhancer"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_LOCAL_DB_URL)

USE_DATABASE = False
DB_STATUS_MESSAGE = "⚠️ Using in-memory fallback"

# In-memory fallback
jobs_db: Dict[str, Dict[str, Any]] = {}
batches_db: Dict[str, Dict[str, Any]] = {}
db_lock = threading.Lock()


# =========================
# DB HELPERS
# =========================
def get_conn():
    if not USE_DATABASE:
        raise RuntimeError("Database is not configured")
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def init_db():
    global USE_DATABASE, DB_STATUS_MESSAGE

    if not PSYCOPG2_AVAILABLE:
        USE_DATABASE = False
        DB_STATUS_MESSAGE = "⚠️ Using in-memory fallback: psycopg2 not installed"
        return

    if not DATABASE_URL:
        USE_DATABASE = False
        DB_STATUS_MESSAGE = "⚠️ Using in-memory fallback: DATABASE_URL missing"
        return

    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                filename TEXT,
                scale INTEGER,
                prompt TEXT,
                style_preset TEXT,
                prompt_config JSONB,
                input_path TEXT,
                output_path TEXT,
                output_name TEXT,
                original_size JSONB,
                enhanced_size JSONB,
                metrics JSONB,
                error TEXT,
                batch_id TEXT,
                created_at DOUBLE PRECISION NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS batches (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress DOUBLE PRECISION NOT NULL DEFAULT 0,
                jobs JSONB,
                zip_path TEXT,
                created_at DOUBLE PRECISION NOT NULL
            )
        """)

        conn.commit()
        cur.close()
        conn.close()

        USE_DATABASE = True
        DB_STATUS_MESSAGE = "✅ PostgreSQL connected"

    except Exception as e:
        USE_DATABASE = False
        DB_STATUS_MESSAGE = f"⚠️ Using in-memory fallback: DB connection failed ({e})"


@app.on_event("startup")
async def startup_event():
    init_db()
    print(DB_STATUS_MESSAGE)


# =========================
# HELPERS
# =========================
def safe_json(value, default=None):
    return default if value is None else value


def cleanup_file(path_value: Optional[str]):
    try:
        if path_value:
            path = Path(path_value)
            if path.exists():
                path.unlink(missing_ok=True)
    except Exception:
        pass


# =========================
# STORAGE LAYER
# =========================
def create_job(
    job_id: str,
    filename: str,
    scale: int,
    input_path: str,
    prompt: str = "",
    style_preset: str = "auto",
    prompt_config: Optional[Dict[str, Any]] = None,
    batch_id: Optional[str] = None,
):
    job_data = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "filename": filename,
        "scale": scale,
        "prompt": prompt,
        "style_preset": style_preset,
        "prompt_config": prompt_config or {},
        "input_path": input_path,
        "output_path": None,
        "output_name": None,
        "original_size": None,
        "enhanced_size": None,
        "metrics": None,
        "error": None,
        "batch_id": batch_id,
        "created_at": time.time(),
    }

    if USE_DATABASE:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO jobs (
                id, status, progress, filename, scale, prompt, style_preset,
                prompt_config, input_path, output_path, output_name,
                original_size, enhanced_size, metrics, error, batch_id, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            job_id,
            "queued",
            0,
            filename,
            scale,
            prompt,
            style_preset,
            psycopg2.extras.Json(prompt_config or {}),
            input_path,
            None,
            None,
            None,
            None,
            None,
            None,
            batch_id,
            time.time(),
        ))
        conn.commit()
        cur.close()
        conn.close()
    else:
        with db_lock:
            jobs_db[job_id] = job_data


def update_job(job_id: str, **fields):
    if not fields:
        return

    if USE_DATABASE:
        json_fields = {"prompt_config", "original_size", "enhanced_size", "metrics"}
        set_parts = []
        values = []

        for key, value in fields.items():
            set_parts.append(f"{key}=%s")
            if key in json_fields and value is not None:
                values.append(psycopg2.extras.Json(value))
            else:
                values.append(value)

        values.append(job_id)

        conn = get_conn()
        cur = conn.cursor()
        cur.execute(f"UPDATE jobs SET {', '.join(set_parts)} WHERE id=%s", values)
        conn.commit()
        cur.close()
        conn.close()
    else:
        with db_lock:
            if job_id in jobs_db:
                jobs_db[job_id].update(fields)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    if USE_DATABASE:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT * FROM jobs WHERE id=%s", (job_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return dict(row) if row else None

    with db_lock:
        return jobs_db.get(job_id)


def list_recent_jobs(limit: int = 10) -> List[Dict[str, Any]]:
    if USE_DATABASE:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT %s", (limit,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(row) for row in rows]

    with db_lock:
        return sorted(jobs_db.values(), key=lambda j: j["created_at"], reverse=True)[:limit]


def create_batch(batch_id: str, job_ids: List[str]):
    batch_data = {
        "id": batch_id,
        "status": "processing",
        "progress": 0.0,
        "jobs": job_ids,
        "zip_path": None,
        "created_at": time.time(),
    }

    if USE_DATABASE:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO batches (id, status, progress, jobs, zip_path, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            batch_id,
            "processing",
            0.0,
            psycopg2.extras.Json(job_ids),
            None,
            time.time(),
        ))
        conn.commit()
        cur.close()
        conn.close()
    else:
        with db_lock:
            batches_db[batch_id] = batch_data


def update_batch(batch_id: str, **fields):
    if not fields:
        return

    if USE_DATABASE:
        set_parts = []
        values = []

        for key, value in fields.items():
            set_parts.append(f"{key}=%s")
            if key == "jobs" and value is not None:
                values.append(psycopg2.extras.Json(value))
            else:
                values.append(value)

        values.append(batch_id)

        conn = get_conn()
        cur = conn.cursor()
        cur.execute(f"UPDATE batches SET {', '.join(set_parts)} WHERE id=%s", values)
        conn.commit()
        cur.close()
        conn.close()
    else:
        with db_lock:
            if batch_id in batches_db:
                batches_db[batch_id].update(fields)


def get_batch(batch_id: str) -> Optional[Dict[str, Any]]:
    if USE_DATABASE:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT * FROM batches WHERE id=%s", (batch_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return dict(row) if row else None

    with db_lock:
        return batches_db.get(batch_id)


def list_recent_batches(limit: int = 5) -> List[Dict[str, Any]]:
    if USE_DATABASE:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT * FROM batches ORDER BY created_at DESC LIMIT %s", (limit,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(row) for row in rows]

    with db_lock:
        return sorted(batches_db.values(), key=lambda b: b["created_at"], reverse=True)[:limit]


def delete_old_data():
    cutoff = time.time() - CLEANUP_INTERVAL

    old_output_paths = []
    old_zip_paths = []

    if USE_DATABASE:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute("SELECT output_path FROM jobs WHERE created_at < %s", (cutoff,))
        old_output_paths = [row["output_path"] for row in cur.fetchall() if row.get("output_path")]

        cur.execute("SELECT zip_path FROM batches WHERE created_at < %s", (cutoff,))
        old_zip_paths = [row["zip_path"] for row in cur.fetchall() if row.get("zip_path")]

        cur.execute("DELETE FROM jobs WHERE created_at < %s", (cutoff,))
        cur.execute("DELETE FROM batches WHERE created_at < %s", (cutoff,))

        conn.commit()
        cur.close()
        conn.close()
    else:
        with db_lock:
            old_job_ids = [jid for jid, job in jobs_db.items() if job["created_at"] < cutoff]
            old_batch_ids = [bid for bid, batch in batches_db.items() if batch["created_at"] < cutoff]

            for jid in old_job_ids:
                output_path = jobs_db[jid].get("output_path")
                if output_path:
                    old_output_paths.append(output_path)
                del jobs_db[jid]

            for bid in old_batch_ids:
                zip_path = batches_db[bid].get("zip_path")
                if zip_path:
                    old_zip_paths.append(zip_path)
                del batches_db[bid]

    for path_value in old_output_paths:
        cleanup_file(path_value)

    for path_value in old_zip_paths:
        cleanup_file(path_value)


# =========================
# PROMPT LOGIC
# =========================
def parse_prompt(prompt: str, style_preset: str = "auto") -> Dict[str, Any]:
    text = (prompt or "").lower()

    config = {
        "denoise_strength": 1.0,
        "sharpen_strength": 1.0,
        "contrast_strength": 1.0,
        "detail_strength": 1.0,
        "saturation_strength": 1.0,
        "brightness_shift": 0,
        "studio_clean": False,
        "ecommerce": False,
        "soft_look": False,
        "high_detail": False,
        "natural": False,
    }

    preset = (style_preset or "auto").lower()

    if preset == "product":
        config.update({
            "denoise_strength": 1.2,
            "sharpen_strength": 1.25,
            "contrast_strength": 1.08,
            "detail_strength": 1.15,
            "studio_clean": True,
            "ecommerce": True,
        })
    elif preset == "portrait":
        config.update({
            "denoise_strength": 1.1,
            "sharpen_strength": 0.9,
            "contrast_strength": 1.03,
            "soft_look": True,
        })
    elif preset == "document":
        config.update({
            "denoise_strength": 0.8,
            "sharpen_strength": 1.3,
            "contrast_strength": 1.2,
            "detail_strength": 1.1,
            "saturation_strength": 0.9,
        })
    elif preset == "render":
        config.update({
            "denoise_strength": 1.1,
            "sharpen_strength": 1.25,
            "contrast_strength": 1.1,
            "detail_strength": 1.2,
            "studio_clean": True,
        })

    if "sharp" in text or "sharper" in text or "crispy" in text:
        config["sharpen_strength"] += 0.25

    if "very sharp" in text or "ultra sharp" in text:
        config["sharpen_strength"] += 0.25

    if "denoise" in text or "remove noise" in text or "clean" in text:
        config["denoise_strength"] += 0.25

    if "high detail" in text or "more detail" in text or "detailed" in text:
        config["detail_strength"] += 0.25
        config["high_detail"] = True

    if "soft" in text or "soft studio" in text:
        config["soft_look"] = True
        config["sharpen_strength"] -= 0.15

    if "natural" in text or "realistic" in text:
        config["natural"] = True
        config["contrast_strength"] = min(config["contrast_strength"], 1.05)
        config["saturation_strength"] = min(config["saturation_strength"], 1.02)

    if "bright" in text:
        config["brightness_shift"] += 8

    if "darker" in text:
        config["brightness_shift"] -= 8

    if "vivid" in text or "vibrant" in text:
        config["saturation_strength"] += 0.15

    if "e-commerce" in text or "ecommerce" in text or "marketplace" in text:
        config["ecommerce"] = True
        config["studio_clean"] = True
        config["contrast_strength"] += 0.05

    if "studio" in text:
        config["studio_clean"] = True

    return config


# =========================
# IMAGE PROCESSING
# =========================
def safe_scale_for_memory(width: int, height: int, scale: int) -> int:
    pixels = width * height
    if scale == 4 and pixels > 2_000_000:
        return 2
    return scale


def safe_read_image(input_path: str) -> np.ndarray:
    img = cv2.imread(input_path)
    if img is None:
        pil_img = Image.open(input_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img


def adjust_saturation_brightness(
    image: np.ndarray,
    saturation: float = 1.0,
    brightness_shift: int = 0,
) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + brightness_shift, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def enhance_image(
    img_array: np.ndarray,
    scale: int,
    prompt_config: Dict[str, Any],
    original_img: Optional[np.ndarray] = None,
):
    denoise_strength = float(prompt_config.get("denoise_strength", 1.0))
    sharpen_strength = float(prompt_config.get("sharpen_strength", 1.0))
    contrast_strength = float(prompt_config.get("contrast_strength", 1.0))
    detail_strength = float(prompt_config.get("detail_strength", 1.0))
    saturation_strength = float(prompt_config.get("saturation_strength", 1.0))
    brightness_shift = int(prompt_config.get("brightness_shift", 0))
    soft_look = bool(prompt_config.get("soft_look", False))
    studio_clean = bool(prompt_config.get("studio_clean", False))

    d = max(5, int(9 * denoise_strength))
    sigma_color = max(40, int(75 * denoise_strength))
    sigma_space = max(40, int(75 * denoise_strength))
    denoised = cv2.bilateralFilter(img_array, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    h, w = denoised.shape[:2]
    interpolation = cv2.INTER_CUBIC if scale == 4 else cv2.INTER_LANCZOS4
    upscaled = cv2.resize(denoised, (w * scale, h * scale), interpolation=interpolation)

    blur_sigma = 2.0 if soft_look else 3.0
    blur = cv2.GaussianBlur(upscaled, (0, 0), blur_sigma)
    alpha = 1.0 + (0.3 * sharpen_strength)
    beta = -(0.3 * sharpen_strength)
    sharpened = cv2.addWeighted(upscaled, alpha, blur, beta, 0)

    clip_limit = max(1.0, min(4.0, 2.0 * contrast_strength))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    sigma_s = max(5, int(10 * detail_strength))
    sigma_r = min(0.4, max(0.1, 0.15 * detail_strength))
    detailed = cv2.detailEnhance(enhanced, sigma_s=sigma_s, sigma_r=sigma_r)

    guided = None
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        try:
            guided = cv2.ximgproc.guidedFilter(
                guide=detailed,
                src=detailed,
                radius=4,
                eps=1e-2,
            )
        except Exception:
            guided = None

    result = guided if guided is not None else detailed
    result = adjust_saturation_brightness(
        result,
        saturation=saturation_strength,
        brightness_shift=brightness_shift,
    )

    if studio_clean:
        result = cv2.fastNlMeansDenoisingColored(result, None, 3, 3, 7, 21)

    metrics = {}
    if original_img is not None:
        try:
            metrics = compute_metrics(original_img, result)
        except Exception:
            metrics = {}

    return result, metrics


# =========================
# WORKERS
# =========================
def process_job(job_id: str, input_path: str, scale: int, original_name: str):
    try:
        job = get_job(job_id)
        if not job:
            return

        update_job(job_id, status="processing", progress=10)

        valid, err = validate_image(input_path)
        if not valid:
            update_job(job_id, status="error", progress=100, error=f"Validation failed: {err}")
            return

        img = safe_read_image(input_path)
        original_img = img.copy()
        update_job(job_id, progress=25)

        safe_scale = safe_scale_for_memory(img.shape[1], img.shape[0], scale)

        prompt = job.get("prompt") or ""
        style_preset = job.get("style_preset") or "auto"
        prompt_config = parse_prompt(prompt, style_preset)

        update_job(job_id, progress=50, scale=safe_scale, prompt_config=prompt_config)

        enhanced, metrics = enhance_image(
            img_array=img,
            scale=safe_scale,
            prompt_config=prompt_config,
            original_img=original_img,
        )

        update_job(job_id, progress=85)

        stem = Path(original_name).stem
        out_name = f"{stem}_{safe_scale}x_enhanced.png"
        out_path = OUTPUT_DIR / f"{job_id}_{out_name}"

        success = cv2.imwrite(str(out_path), enhanced, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not success:
            raise RuntimeError("Failed to save enhanced image")

        update_job(
            job_id,
            status="done",
            progress=100,
            output_path=str(out_path),
            output_name=out_name,
            original_size=[original_img.shape[1], original_img.shape[0]],
            enhanced_size=[enhanced.shape[1], enhanced.shape[0]],
            metrics=metrics,
            error=None,
        )

    except Exception as e:
        update_job(job_id, status="error", progress=100, error=str(e))


def update_batch_progress(batch_id: str):
    batch = get_batch(batch_id)
    if not batch:
        return

    job_ids = batch.get("jobs") or []
    total = len(job_ids)
    if total == 0:
        update_batch(batch_id, progress=100, status="done")
        return

    done_count = 0
    error_count = 0
    successful_outputs = []
    successful_names = []

    for jid in job_ids:
        job = get_job(jid)
        if not job:
            continue

        if job["status"] == "done":
            done_count += 1
            if job.get("output_path") and Path(job["output_path"]).exists():
                successful_outputs.append(Path(job["output_path"]))
                successful_names.append(job.get("filename") or "output.png")
        elif job["status"] == "error":
            error_count += 1

    completed = done_count + error_count
    progress = round((completed / total) * 100, 2)

    if completed == total:
        zip_path = None
        if successful_outputs:
            zip_path = str(create_zip_batch(batch_id, successful_outputs, successful_names))
        update_batch(batch_id, progress=progress, status="done", zip_path=zip_path)
    else:
        update_batch(batch_id, progress=progress)


def process_batch_job(batch_id: str, job_id: str):
    try:
        job = get_job(job_id)
        if not job:
            return
        process_job(job_id, job["input_path"], job["scale"], job["filename"])
    finally:
        update_batch_progress(batch_id)


# =========================
# API
# =========================
@app.post("/api/enhance")
async def enhance(
    file: UploadFile = File(...),
    scale: int = Form(2),
    prompt: str = Form(""),
    style_preset: str = Form("auto"),
):
    if scale not in (2, 4):
        raise HTTPException(400, "Scale must be 2 or 4")

    allowed_presets = {"auto", "product", "portrait", "document", "render"}
    if style_preset not in allowed_presets:
        raise HTTPException(400, f"style_preset must be one of {sorted(allowed_presets)}")

    job_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix or ".png"
    input_path = UPLOAD_DIR / f"{job_id}{ext}"

    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    create_job(
        job_id=job_id,
        filename=file.filename,
        scale=scale,
        input_path=str(input_path),
        prompt=prompt,
        style_preset=style_preset,
        prompt_config=parse_prompt(prompt, style_preset),
    )

    delete_old_data()
    BATCH_EXECUTOR.submit(process_job, job_id, str(input_path), scale, file.filename)

    return {
        "job_id": job_id,
        "message": "Enhancement job created",
        "prompt": prompt,
        "style_preset": style_preset,
    }


@app.post("/api/enhance-batch")
async def enhance_batch(
    files: List[UploadFile] = File(...),
    scale: int = Form(2),
    prompt: str = Form(""),
    style_preset: str = Form("auto"),
):
    if scale not in (2, 4):
        raise HTTPException(400, "Scale must be 2 or 4")
    if len(files) > 10:
        raise HTTPException(400, "Max 10 files per batch")

    allowed_presets = {"auto", "product", "portrait", "document", "render"}
    if style_preset not in allowed_presets:
        raise HTTPException(400, f"style_preset must be one of {sorted(allowed_presets)}")

    batch_id = str(uuid.uuid4())
    batch_job_ids = []

    for file in files:
        job_id = str(uuid.uuid4())
        ext = Path(file.filename).suffix or ".png"
        input_path = UPLOAD_DIR / f"{job_id}{ext}"

        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        create_job(
            job_id=job_id,
            filename=file.filename,
            scale=scale,
            input_path=str(input_path),
            prompt=prompt,
            style_preset=style_preset,
            prompt_config=parse_prompt(prompt, style_preset),
            batch_id=batch_id,
        )
        batch_job_ids.append(job_id)

    create_batch(batch_id, batch_job_ids)
    delete_old_data()

    for job_id in batch_job_ids:
        BATCH_EXECUTOR.submit(process_batch_job, batch_id, job_id)

    return {
        "batch_id": batch_id,
        "job_count": len(files),
        "prompt": prompt,
        "style_preset": style_preset,
    }


@app.get("/api/job/{job_id}")
async def api_get_job(job_id: str):
    delete_old_data()
    job = get_job(job_id)

    if not job:
        raise HTTPException(404, "Job not found")

    return {
        "id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "filename": job["filename"],
        "scale": job["scale"],
        "prompt": job.get("prompt"),
        "style_preset": job.get("style_preset"),
        "prompt_config": safe_json(job.get("prompt_config"), {}),
        "original_size": safe_json(job.get("original_size")),
        "enhanced_size": safe_json(job.get("enhanced_size")),
        "output_name": job.get("output_name"),
        "metrics": safe_json(job.get("metrics"), {}),
        "error": job.get("error"),

        # ✅ 🔥 IMPORTANT FIX
        "image_url": f"/api/download/{job['id']}" if job["status"] == "done" else None
    }

@app.get("/api/batch/{batch_id}")
async def api_get_batch(batch_id: str):
    delete_old_data()
    batch = get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")

    jobs_list = []
    for jid in batch.get("jobs") or []:
        job = get_job(jid)
        if job:
            jobs_list.append({
                "id": job["id"],
                "status": job["status"],
                "progress": job["progress"],
                "filename": job["filename"],
                "scale": job["scale"],
                "prompt": job.get("prompt"),
                "style_preset": job.get("style_preset"),
                "metrics": safe_json(job.get("metrics"), {}),
                "error": job.get("error"),
            })

    return {
        "batch_id": batch["id"],
        "status": batch["status"],
        "progress": batch["progress"],
        "job_count": len(batch.get("jobs") or []),
        "jobs": jobs_list,
        "zip_path": batch.get("zip_path"),
    }


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    delete_old_data()
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done":
        raise HTTPException(400, "Job not complete")
    if not job.get("output_path") or not Path(job["output_path"]).exists():
        raise HTTPException(404, "Output file not found")

    return FileResponse(job["output_path"], filename=job["output_name"], media_type="image/png")


@app.get("/api/batch-download/{batch_id}")
async def batch_download(batch_id: str):
    delete_old_data()
    batch = get_batch(batch_id)
    if not batch or batch["status"] != "done":
        raise HTTPException(400, "Batch not complete")
    if not batch.get("zip_path") or not Path(batch["zip_path"]).exists():
        raise HTTPException(400, "No completed files available for zip download")

    return FileResponse(
        batch["zip_path"],
        filename=f"batch_{batch_id}_enhanced.zip",
        media_type="application/zip",
    )


@app.get("/api/preview/{job_id}")
async def preview(job_id: str):
    delete_old_data()
    job = get_job(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(400, "Not ready")
    if not job.get("output_path") or not Path(job["output_path"]).exists():
        raise HTTPException(404, "Output file not found")

    with open(job["output_path"], "rb") as f:
        data = base64.b64encode(f.read()).decode()

    return {
        "image": f"data:image/png;base64,{data}",
        "prompt": job.get("prompt"),
        "style_preset": job.get("style_preset"),
    }


@app.get("/api/jobs")
async def api_list_jobs():
    delete_old_data()
    jobs = list_recent_jobs(10)

    return [
        {
            "id": job["id"],
            "status": job["status"],
            "progress": job["progress"],
            "filename": job["filename"],
            "scale": job["scale"],
            "prompt": job.get("prompt"),
            "style_preset": job.get("style_preset"),
            "metrics": safe_json(job.get("metrics"), {}),
            "error": job.get("error"),
            "created_at": job["created_at"],
        }
        for job in jobs
    ]


@app.get("/api/batches")
async def api_list_batches():
    delete_old_data()
    batches = list_recent_batches(5)

    return [
        {
            "id": batch["id"],
            "status": batch["status"],
            "progress": batch["progress"],
            "job_count": len(batch.get("jobs") or []),
            "created_at": batch["created_at"],
        }
        for batch in batches
    ]


@app.get("/api/prompt-presets")
async def prompt_presets():
    return {
        "style_presets": [
            {"id": "auto", "label": "Auto", "description": "General balanced enhancement"},
            {"id": "product", "label": "Product", "description": "Cleaner product visuals, sharper edges, marketplace-ready"},
            {"id": "portrait", "label": "Portrait", "description": "Softer skin-friendly look with moderate sharpening"},
            {"id": "document", "label": "Document", "description": "Text clarity and contrast boost"},
            {"id": "render", "label": "Render", "description": "Sharper product renders and concept visuals"},
        ],
        "example_prompts": [
            "make it sharper and cleaner",
            "clean product render with high detail",
            "e-commerce ready white-background look",
            "soft studio look but keep details",
            "natural realistic enhancement",
            "remove noise and improve clarity",
        ],
    }


# =========================
# STATIC ROUTES
# =========================
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_path = STATIC_DIR / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(favicon_path)
    raise HTTPException(status_code=404, detail="Favicon not found")


@app.get("/", response_class=HTMLResponse)
async def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "<h1>UI not found</h1>"


app.mount("/static", StaticFiles(directory="static"), name="static")
