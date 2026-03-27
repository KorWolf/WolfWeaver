from datetime import datetime
from pathlib import Path
import re
import threading
import uuid

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from src.config_loader import load_config
from src.palette import load_palette_from_json
from src.palette_presets import PALETTE_PRESETS
from src.pipeline import run_pipeline_stream


HEX_COLOR_PATTERN = re.compile(r"^#?[0-9A-Fa-f]{6}$")
SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
RUNS_ROOT = Path("output/runs").resolve()
UPLOAD_ROOT = Path("input/uploads")
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def load_default_palette_text() -> str:
    defaults = load_config(Path("config.json"))
    palette_data = load_palette_from_json(Path(defaults["palette_file"]))
    return "\n".join(str(color).upper() for color in palette_data["colors"])


def build_default_form_values() -> dict:
    defaults = load_config(Path("config.json"))
    return {
        "frame_count": int(defaults["frame_count"]),
        "reconstruction_mode": str(defaults["reconstruction_mode"]),
        "random_seed": int(defaults["random_seed"]),
        "create_gif": bool(defaults["create_gif"]),
        "gif_frame_duration_ms": int(defaults["gif_frame_duration_ms"]),
        "palette_text": load_default_palette_text(),
        "frame_prefix": str(defaults["frame_prefix"]),
        "gif_output_name": str(defaults["gif_output_name"]),
        "selected_preset": "",
    }


def parse_palette_text(palette_text: str) -> list[str]:
    lines = [line.strip() for line in palette_text.splitlines()]
    lines = [line for line in lines if line]

    if not lines:
        raise ValueError("Palette cannot be empty.")

    normalized_colors: list[str] = []

    for index, line in enumerate(lines, start=1):
        if not HEX_COLOR_PATTERN.fullmatch(line):
            raise ValueError(
                f"Invalid hex color on line {index}: {line}. "
                f"Use one 6-digit hex value per line, such as #00A94F."
            )

        normalized = line.upper()
        if not normalized.startswith("#"):
            normalized = f"#{normalized}"

        normalized_colors.append(normalized)

    return normalized_colors


def validate_positive_int(value: str, field_name: str, minimum: int = 1) -> int:
    try:
        int_value = int(value)
    except Exception as error:
        raise ValueError(f"{field_name} must be a whole number.") from error

    if int_value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")

    return int_value


def validate_safe_name(value: str, field_name: str) -> str:
    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{field_name} cannot be empty.")

    if not SAFE_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"{field_name} contains invalid characters. "
            f"Use only letters, numbers, dots, underscores, and hyphens."
        )

    return normalized


def validate_image_upload(file_storage) -> Path | None:
    if file_storage is None or not file_storage.filename:
        return None

    filename = secure_filename(file_storage.filename)
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError("Unsupported image file type. Use PNG, JPG, JPEG, or WEBP.")

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_name = f"{timestamp}_{filename}"
    saved_path = UPLOAD_ROOT / saved_name
    file_storage.save(saved_path)

    return saved_path


def build_frame_count_message(frame_count: int, palette_size: int) -> str | None:
    if frame_count == palette_size:
        return None

    if frame_count > palette_size:
        return (
            f"Some frames may repeat because frame count ({frame_count}) exceeds "
            f"palette size ({palette_size})."
        )

    return (
        f"Not all possible frame iterations will be generated because frame count "
        f"({frame_count}) is smaller than palette size ({palette_size})."
    )


def build_runtime_text(total_runtime_seconds: float) -> str:
    minutes = int(total_runtime_seconds // 60)
    seconds = total_runtime_seconds % 60
    return f"{minutes}m {seconds:.2f}s"


def build_config_from_request() -> tuple[dict, dict, str | None]:
    defaults = load_config(Path("config.json"))

    form_values = {
        "frame_count": request.form.get("frame_count", str(defaults["frame_count"])),
        "reconstruction_mode": request.form.get(
            "reconstruction_mode",
            str(defaults["reconstruction_mode"]),
        ),
        "random_seed": request.form.get("random_seed", str(defaults["random_seed"])),
        "create_gif": "create_gif" in request.form,
        "gif_frame_duration_ms": request.form.get(
            "gif_frame_duration_ms",
            str(defaults["gif_frame_duration_ms"]),
        ),
        "palette_text": request.form.get("palette_text", load_default_palette_text()),
        "frame_prefix": request.form.get("frame_prefix", str(defaults["frame_prefix"])),
        "gif_output_name": request.form.get(
            "gif_output_name",
            str(defaults["gif_output_name"]),
        ),
        "selected_preset": request.form.get("selected_preset", ""),
    }

    palette_colors = parse_palette_text(form_values["palette_text"])
    palette_size = len(palette_colors)

    frame_count = validate_positive_int(form_values["frame_count"], "Frame count", minimum=1)
    random_seed = validate_positive_int(form_values["random_seed"], "Random seed", minimum=0)
    gif_frame_duration_ms = validate_positive_int(
        form_values["gif_frame_duration_ms"],
        "GIF frame duration",
        minimum=1,
    )

    reconstruction_mode = form_values["reconstruction_mode"]
    if reconstruction_mode not in {
        "scanline",
        "random_seeded",
        "random_unseeded",
        "weighted_random",
    }:
        raise ValueError("Invalid reconstruction mode.")

    frame_prefix = validate_safe_name(form_values["frame_prefix"], "Frame prefix")
    gif_output_name = validate_safe_name(form_values["gif_output_name"], "GIF output name")

    if not gif_output_name.lower().endswith(".gif"):
        gif_output_name = f"{gif_output_name}.gif"

    uploaded_image_path = validate_image_upload(request.files.get("source_image"))
    source_image = uploaded_image_path if uploaded_image_path else Path(defaults["source_image"])

    if not source_image.exists():
        raise ValueError("No valid source image was provided, and the default image was not found.")

    config = {
        "source_image": str(source_image),
        "palette_file": str(defaults["palette_file"]),
        "palette_colors": palette_colors,
        "palette_name": form_values["selected_preset"] or "web_palette",
        "frame_count": frame_count,
        "save_debug_tables": False,
        "create_gif": bool(form_values["create_gif"]),
        "gif_frame_duration_ms": gif_frame_duration_ms,
        "gif_output_name": gif_output_name,
        "frame_prefix": frame_prefix,
        "reconstruction_mode": reconstruction_mode,
        "random_seed": random_seed,
    }

    advisory = build_frame_count_message(frame_count, palette_size)
    return config, form_values, advisory


def run_file_url(path: Path) -> str:
    relative_path = path.resolve().relative_to(RUNS_ROOT)
    return url_for("serve_run_file", subpath=relative_path.as_posix())


def serialize_update(update: dict) -> dict:
    frame_urls = [
        run_file_url(path)
        for path in update.get("frame_paths", [])
    ]

    gif_url = None
    gif_path = update.get("gif_path")
    if gif_path is not None:
        gif_url = run_file_url(gif_path)

    total_runtime_seconds = float(update.get("total_runtime_seconds", 0.0))
    frame_count = int(update.get("frame_count", 0))
    completed_frames = int(update.get("completed_frames", 0))

    percent_complete = 0
    if frame_count > 0:
        percent_complete = int((completed_frames / frame_count) * 100)

    if update.get("status") == "completed":
        percent_complete = 100

    return {
        "status": update.get("status"),
        "run_dir": str(update.get("run_dir", "")),
        "config_snapshot_path": str(update.get("config_snapshot_path", "")),
        "source_stats_csv": str(update.get("source_stats_csv", "")),
        "image_width": update.get("image_width"),
        "image_height": update.get("image_height"),
        "total_pixels": update.get("total_pixels"),
        "unique_source_colors": update.get("unique_source_colors"),
        "frame_count": frame_count,
        "completed_frames": completed_frames,
        "percent_complete": percent_complete,
        "total_runtime_seconds": total_runtime_seconds,
        "runtime_text": build_runtime_text(total_runtime_seconds),
        "palette_name": update.get("palette_name"),
        "palette_size": update.get("palette_size"),
        "reconstruction_mode": update.get("reconstruction_mode"),
        "frame_urls": frame_urls,
        "gif_url": gif_url,
    }


def worker(job_id: str, config: dict) -> None:
    try:
        for update in run_pipeline_stream(config):
            with jobs_lock:
                jobs[job_id]["update"] = update
                jobs[job_id]["status"] = update["status"]
    except Exception as error:
        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(error)


@app.route("/", methods=["GET"])
def form_page():
    form_values = build_default_form_values()

    try:
        advisory = build_frame_count_message(
            frame_count=int(form_values["frame_count"]),
            palette_size=len(parse_palette_text(form_values["palette_text"])),
        )
    except Exception:
        advisory = None

    return render_template(
        "form.html",
        error=None,
        advisory=advisory,
        values=form_values,
        palette_presets=PALETTE_PRESETS,
    )


@app.route("/start", methods=["POST"])
def start_run():
    try:
        config, form_values, advisory = build_config_from_request()
        job_id = uuid.uuid4().hex

        with jobs_lock:
            jobs[job_id] = {
                "status": "queued",
                "update": None,
                "error": None,
                "advisory": advisory,
            }

        thread = threading.Thread(target=worker, args=(job_id, config), daemon=True)
        thread.start()

        return redirect(url_for("run_page", job_id=job_id))

    except Exception as error:
        defaults = build_default_form_values()
        attempted_values = {
            **defaults,
            "frame_count": request.form.get("frame_count", defaults["frame_count"]),
            "reconstruction_mode": request.form.get(
                "reconstruction_mode",
                defaults["reconstruction_mode"],
            ),
            "random_seed": request.form.get("random_seed", defaults["random_seed"]),
            "create_gif": "create_gif" in request.form,
            "gif_frame_duration_ms": request.form.get(
                "gif_frame_duration_ms",
                defaults["gif_frame_duration_ms"],
            ),
            "palette_text": request.form.get("palette_text", defaults["palette_text"]),
            "frame_prefix": request.form.get("frame_prefix", defaults["frame_prefix"]),
            "gif_output_name": request.form.get("gif_output_name", defaults["gif_output_name"]),
            "selected_preset": request.form.get("selected_preset", ""),
        }

        try:
            advisory = build_frame_count_message(
                frame_count=int(attempted_values["frame_count"]),
                palette_size=len(parse_palette_text(attempted_values["palette_text"])),
            )
        except Exception:
            advisory = None

        return render_template(
            "form.html",
            error=str(error),
            advisory=advisory,
            values=attempted_values,
            palette_presets=PALETTE_PRESETS,
        ), 400


@app.route("/run/<job_id>", methods=["GET"])
def run_page(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        return "Run not found.", 404

    return render_template(
        "run.html",
        job_id=job_id,
        advisory=job.get("advisory"),
    )


@app.route("/api/run/<job_id>", methods=["GET"])
def run_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        return jsonify({"status": "missing", "error": "Run not found."}), 404

    if job["status"] == "failed":
        return jsonify({
            "status": "failed",
            "error": job["error"],
        })

    update = job.get("update")
    if update is None:
        return jsonify({
            "status": "queued",
            "completed_frames": 0,
            "frame_count": 0,
            "percent_complete": 0,
            "runtime_text": "0m 0.00s",
            "frame_urls": [],
            "gif_url": None,
        })

    payload = serialize_update(update)
    payload["advisory"] = job.get("advisory")
    return jsonify(payload)


@app.route("/runs/<path:subpath>", methods=["GET"])
def serve_run_file(subpath: str):
    return send_from_directory(RUNS_ROOT, subpath)


def main() -> None:
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()