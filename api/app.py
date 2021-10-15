
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import API_HOST, API_PORT, DEBUG, OUTPUT_DIR, DOCS_DIR
from core.pipeline import PipelineConfig, ServiceBoundaryPipeline
from ml.boundary_ranker import BoundaryRanker

log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024   # 64 MB upload limit

_ALLOWED_DOC_EXTS = {".pdf", ".docx", ".md", ".txt", ".html"}
_ALLOWED_TRACE_EXTS = {".log", ".csv", ".txt"}

@app.route("/api/v1/upload-doc", methods=["POST"])
def upload_doc():
    if "file" not in request.files:
        return jsonify({"error": "No file field in request"}), 400

    f    = request.files["file"]
    name = secure_filename(f.filename or "upload")
    ext  = Path(name).suffix.lower()

    if ext not in _ALLOWED_DOC_EXTS:
        return jsonify({"error": f"Unsupported extension: {ext}",
                        "allowed": list(_ALLOWED_DOC_EXTS)}), 415

    dest = str(DOCS_DIR / name)
    f.save(dest)
    log.info("Document saved: %s", dest)
    return jsonify({"saved_path": dest, "filename": name}), 201


# ─────────────────────────────────────────────────────────────────────────────
# Trace upload
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/v1/upload-trace", methods=["POST"])
def upload_trace():
    if "file" not in request.files:
        return jsonify({"error": "No file field"}), 400

    f    = request.files["file"]
    name = secure_filename(f.filename or "trace.log")
    ext  = Path(name).suffix.lower()

    if ext not in _ALLOWED_TRACE_EXTS:
        return jsonify({"error": f"Unsupported extension: {ext}",
                        "allowed": list(_ALLOWED_TRACE_EXTS)}), 415

    dest_dir = OUTPUT_DIR / "traces"
    dest_dir.mkdir(exist_ok=True)
    dest = str(dest_dir / name)
    f.save(dest)
    log.info("Trace saved: %s", dest)
    return jsonify({"saved_path": dest, "filename": name}), 201


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )
    log.info("Starting Service Boundary Detector API on %s:%d", API_HOST, API_PORT)
    app.run(host=API_HOST, port=API_PORT, debug=DEBUG)
