import json
import logging
import os
import sys
import traceback
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import API_HOST, API_PORT, DEBUG, OUTPUT_DIR, DOCS_DIR
from core.pipeline import PipelineConfig, ServiceBoundaryPipeline
from ml.boundary_ranker import BoundaryRanker
from feedback.feedback_store import FeedbackStore

log = logging.getLogger(__name__)

app  = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

# Single shared feedback store instance (persists to output/feedback_store.json)
_feedback_store = FeedbackStore()

_ALLOWED_DOC_EXTS   = {".pdf", ".docx", ".md", ".txt", ".html"}
_ALLOWED_TRACE_EXTS = {".log", ".csv", ".txt"}

UI_HTML = Path(__file__).parent.parent / "ui" / "review.html"

@app.route("/", methods=["GET"])
def serve_ui():
    if UI_HTML.exists():
        return send_file(str(UI_HTML))
    return (
        "<h2>UI not found.</h2>"
        "<p>Make sure <code>ui/review.html</code> exists.</p>"
        "<p>API: <a href='/api/v1/health'>/api/v1/health</a></p>",
        404,
    )


@app.route("/api/v1/health", methods=["GET"])
def health():
    model_path = Path(__file__).parent.parent / "models" / "boundary_ranker.joblib"
    return jsonify({
        "status":           "ok",
        "service":          "ServiceBoundaryDetector",
        "version":          "1.0.0",
        "feedback_stats":   _feedback_store.stats(),
        "model_trained":    model_path.exists(),
        "ui_url":           f"http://{API_HOST}:{API_PORT}/",
    })



@app.route("/api/v1/analyze", methods=["POST"])
def analyze():
    body       = request.get_json(force=True, silent=True) or {}
    code_roots = body.get("code_roots")
    if not code_roots:
        return jsonify({"error": "'code_roots' is required"}), 400
    for root in code_roots:
        if not Path(root).exists():
            return jsonify({"error": f"Path does not exist: {root}"}), 400

    stored_labels   = _feedback_store.get_labelled_pairs_for_training()
    labels_file     = body.get("labels_file")
    feedback_injected = 0

    if stored_labels:
        tmp_labels = OUTPUT_DIR / "auto_labels.json"
        with open(tmp_labels, "w") as f:
            json.dump(stored_labels, f)
        labels_file       = str(tmp_labels)
        feedback_injected = len(stored_labels)
        log.info("Auto-injecting %d feedback labels into pipeline", feedback_injected)

    cfg = PipelineConfig(
        code_roots        = code_roots,
        trace_files       = body.get("trace_files")  or [],
        doc_files         = body.get("doc_files")    or [],
        git_repo_path     = body.get("git_repo_path"),
        labels_file       = labels_file,
        top_k             = int(body.get("top_k", 30)),
        skip_behavioral   = bool(body.get("skip_behavioral",   False)),
        skip_evolutionary = bool(body.get("skip_evolutionary", False)),
        skip_documents    = bool(body.get("skip_documents",    False)),
    )

    try:
        result = ServiceBoundaryPipeline(cfg).run()
    except Exception as e:
        log.error("Pipeline error:\n%s", traceback.format_exc())
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    return jsonify({
        "meta":                result.meta,
        "top_boundaries":      [c.to_dict() for c in result.top_candidates],
        "service_clusters":    {str(k): v for k, v in result.cluster_map.items()},
        "feature_importances": result.feature_importances,
        "hotspots":            result.hotspots,
        "units_scanned":       len(result.units),
        "feedback_injected":   feedback_injected,
    }), 200



@app.route("/api/v1/report", methods=["GET"])
def get_report():
    report_path = OUTPUT_DIR / "boundary_report.json"
    if not report_path.exists():
        return jsonify({"error": "No report yet. Run /api/v1/analyze first."}), 404
    with open(report_path) as f:
        data = json.load(f)
    fb_labels = _feedback_store.get_all_labels()
    for b in data.get("top_boundaries", []):
        a, c = b["component_a"], b["component_b"]
        key  = (min(a, c), max(a, c))
        b["feedback_label"] = fb_labels.get(key)  # None if not reviewed
    return jsonify(data), 200



@app.route("/api/v1/feedback", methods=["POST"])
def record_feedback():
    body   = request.get_json(force=True, silent=True) or {}
    comp_a = body.get("comp_a")
    comp_b = body.get("comp_b")
    label  = body.get("label")

    if not comp_a or not comp_b:
        return jsonify({"error": "'comp_a' and 'comp_b' are required"}), 400
    if label not in (0, 1, True, False):
        return jsonify({"error": "'label' must be 0 (keep together) or 1 (valid boundary)"}), 400

    entry = _feedback_store.record(
        comp_a         = comp_a,
        comp_b         = comp_b,
        label          = int(label),
        boundary_score = float(body.get("boundary_score", 0.0)),
        confidence     = float(body.get("confidence",     0.0)),
        rationale_text = body.get("rationale_text", ""),
        decided_by     = body.get("decided_by", "architect"),
        run_id         = body.get("run_id", ""),
    )
    return jsonify({
        "status":             "recorded",
        "entry": {
            "comp_a":         entry.comp_a,
            "comp_b":         entry.comp_b,
            "label":          entry.label,
            "decided_at":     entry.decided_at,
            "rationale_text": entry.rationale_text,
        },
        "stats":              _feedback_store.stats(),
        "ready_for_training": _feedback_store.is_sufficient_for_training(),
    }), 201


@app.route("/api/v1/feedback", methods=["GET"])
def list_feedback():
    entries = _feedback_store.get_all_entries()
    return jsonify({
        "stats":              _feedback_store.stats(),
        "entries": [{
            "comp_a":         e.comp_a,
            "comp_b":         e.comp_b,
            "label":          e.label,
            "boundary_score": e.boundary_score,
            "rationale_text": e.rationale_text,
            "decided_by":     e.decided_by,
            "decided_at":     e.decided_at,
        } for e in entries],
        "ready_for_training": _feedback_store.is_sufficient_for_training(),
    }), 200


@app.route("/api/v1/feedback/delete", methods=["POST"])
def delete_feedback():
    body   = request.get_json(force=True, silent=True) or {}
    comp_a = body.get("comp_a")
    comp_b = body.get("comp_b")
    if not comp_a or not comp_b:
        return jsonify({"error": "comp_a and comp_b required"}), 400
    deleted = _feedback_store.delete(comp_a, comp_b)
    return jsonify({"deleted": deleted, "stats": _feedback_store.stats()}), 200


@app.route("/api/v1/feedback/bulk", methods=["POST"])
def bulk_feedback():
    body      = request.get_json(force=True, silent=True) or {}
    decisions = body.get("decisions", [])
    if not decisions:
        return jsonify({"error": "'decisions' list is required"}), 400
    count = _feedback_store.bulk_record(decisions)
    return jsonify({
        "recorded":           count,
        "stats":              _feedback_store.stats(),
        "ready_for_training": _feedback_store.is_sufficient_for_training(),
    }), 201


@app.route("/api/v1/feedback/all", methods=["DELETE"])
def clear_feedback():
    if request.args.get("confirm", "").lower() != "yes":
        return jsonify({"error": "Add ?confirm=yes to clear all feedback"}), 400
    count = _feedback_store.clear_all()
    return jsonify({"deleted": count, "message": "All feedback cleared"}), 200



@app.route("/api/v1/train", methods=["POST"])
def train():
    body   = request.get_json(force=True, silent=True) or {}
    labels = body.get("labels") or _feedback_store.get_labelled_pairs_for_training()

    if len(labels) < 6:
        return jsonify({
            "error":      "Not enough labelled pairs (minimum 6, need both accept + reject)",
            "current":    _feedback_store.stats(),
            "suggestion": "Review more candidates in the UI at http://localhost:5050/",
        }), 400

    from core.models import PairFeatures
    pairs = []
    for item in labels:
        pf       = PairFeatures(comp_a=item["comp_a"], comp_b=item["comp_b"])
        pf.label = int(item["label"])
        pairs.append(pf)

    ranker  = BoundaryRanker()
    metrics = ranker.train(pairs)

    if "error" in metrics:
        return jsonify({"error": metrics["error"], "details": metrics}), 400

    return jsonify({
        "status":         "trained",
        "n_labels":       len(labels),
        "metrics":        metrics,
        "feedback_stats": _feedback_store.stats(),
    }), 200



@app.route("/api/v1/feature-importances", methods=["GET"])
def feature_importances():
    ranker = BoundaryRanker()
    if not ranker.load():
        return jsonify({"error": "No trained model. Use POST /api/v1/train first."}), 404
    return jsonify({"feature_importances": ranker.feature_importances}), 200



@app.route("/api/v1/upload-doc", methods=["POST"])
def upload_doc():
    if "file" not in request.files:
        return jsonify({"error": "No 'file' field"}), 400
    f   = request.files["file"]
    ext = Path(secure_filename(f.filename or "upload")).suffix.lower()
    if ext not in _ALLOWED_DOC_EXTS:
        return jsonify({"error": f"Unsupported extension {ext}",
                        "allowed": list(_ALLOWED_DOC_EXTS)}), 415
    name = secure_filename(f.filename)
    dest = str(DOCS_DIR / name)
    f.save(dest)
    return jsonify({"saved_path": dest, "filename": name}), 201


@app.route("/api/v1/upload-trace", methods=["POST"])
def upload_trace():
    if "file" not in request.files:
        return jsonify({"error": "No 'file' field"}), 400
    f   = request.files["file"]
    ext = Path(secure_filename(f.filename or "trace.log")).suffix.lower()
    if ext not in _ALLOWED_TRACE_EXTS:
        return jsonify({"error": f"Unsupported extension {ext}",
                        "allowed": list(_ALLOWED_TRACE_EXTS)}), 415
    name     = secure_filename(f.filename)
    dest_dir = OUTPUT_DIR / "traces"
    dest_dir.mkdir(exist_ok=True)
    dest = str(dest_dir / name)
    f.save(dest)
    return jsonify({"saved_path": dest, "filename": name}), 201



@app.errorhandler(413)
def too_large(e):    return jsonify({"error": "File too large (max 64 MB)"}), 413

@app.errorhandler(404)
def not_found(e):    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e): return jsonify({"error": "Internal server error"}), 500



if __name__ == "__main__":
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )
    log.info("─" * 55)
    log.info("  Service Boundary Detector")
    log.info("  Review UI  →  http://%s:%d/", API_HOST, API_PORT)
    log.info("  Health     →  http://%s:%d/api/v1/health", API_HOST, API_PORT)
    log.info("─" * 55)
    app.run(host=API_HOST, port=API_PORT, debug=DEBUG)
