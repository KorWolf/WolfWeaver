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
from src.palette_source import (
    extract_clustered_main_palette_colors,
    extract_top_image_palette_colors,
)
from src.pipeline import run_pipeline_stream
from src.reconstruction_modes import RECONSTRUCTION_MODES, get_reconstruction_mode_values
from src.score_modes import get_score_mode_values


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

# In-memory job state for active runs.
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def load_default_palette_text() -> str:
    """
    Load the default palette file from config.json and convert it to
    textarea-friendly text, one hex color per line.
    """
    defaults = load_config(Path("config.json"))
    palette_data = load_palette_from_json(Path(defaults["palette_file"]))
    return "\n".join(str(color).upper() for color in palette_data["colors"])


def build_default_form_values() -> dict:
    """
    Build default values for the main form from config.json.
    """
    defaults = load_config(Path("config.json"))
    default_palette_text = load_default_palette_text()
    default_palette_size = len(parse_palette_text(default_palette_text))

    return {
        "palette_source": "manual",
        "image_palette_method": "clustered_main",
        "image_palette_count": default_palette_size,
        "frame_count": int(defaults["frame_count"]),
        "reconstruction_mode": str(defaults["reconstruction_mode"]),
        "score_mode": str(defaults["score_mode"]),
        "random_seed": int(defaults["random_seed"]),
        "create_gif": bool(defaults["create_gif"]),
        "gif_frame_duration_ms": int(defaults["gif_frame_duration_ms"]),
        "palette_text": default_palette_text,
        "frame_prefix": str(defaults["frame_prefix"]),
        "gif_output_name": str(defaults["gif_output_name"]),
        "selected_preset": "",
    }


def build_palette_preset_display_options() -> list[dict]:
    """
    Build display labels for preset palettes.

    If the preset key already ends with its color count, do not append
    the count again.
    """
    display_options: list[dict] = []

    for preset_key, preset_colors in PALETTE_PRESETS.items():
        color_count = len(preset_colors)
        base_label = preset_key.replace("_", " ").title()

        already_ends_with_count = re.search(rf"_{color_count}$", preset_key) is not None

        if already_ends_with_count:
            display_label = base_label
        else:
            display_label = f"{base_label} {color_count}"

        display_options.append(
            {
                "value": preset_key,
                "label": display_label,
                "color_count": color_count,
            }
        )

    return display_options


def build_score_mode_options() -> list[dict]:
    """
    Build display labels and descriptions for score modes shown in the UI.
    """
    return [
        {
            "value": "basic_rgb_sl",
            "label": "Basic RGB / S / L",
            "description": "Baseline score using RGB difference plus saturation and lightness difference.",
        },
        {
            "value": "weighted_rgb_sl",
            "label": "Weighted RGB / S / L",
            "description": "Weighted version of the baseline score that gives more influence to perceptually weighted RGB difference.",
        },
        {
            "value": "accent_aware",
            "label": "Accent aware",
            "description": "Adds hue separation and stronger saturation protection so vivid accents like eyes, tongues, and colored details are less likely to be washed out.",
        },
        {
            "value": "separation_aware",
            "label": "Separation aware",
            "description": "Pushes nearby color families apart more aggressively, which can help reduce yellow/beige drift across hair, skin, cloth, and warm accents.",
        },
    ]


def parse_palette_text(palette_text: str) -> list[str]:
    """
    Parse the palette textarea into a normalized list of hex colors.
    """
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
    """
    Validate that a submitted value is an integer >= minimum.
    """
    try:
        int_value = int(value)
    except Exception as error:
        raise ValueError(f"{field_name} must be a whole number.") from error

    if int_value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")

    return int_value


def validate_safe_name(value: str, field_name: str) -> str:
    """
    Validate names used for output files.
    """
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
    """
    Validate and save an uploaded image file.
    """
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
    """
    Build a user-facing advisory about frame count vs palette size.
    """
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
    """
    Format total runtime in minutes and seconds.
    """
    minutes = int(total_runtime_seconds // 60)
    seconds = total_runtime_seconds % 60
    return f"{minutes}m {seconds:.2f}s"


def build_duration_text(total_seconds: float) -> str:
    """
    Format a duration in minutes and seconds.
    """
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes}m {seconds:.2f}s"


def build_config_from_request() -> tuple[dict, dict, str | None]:
    """
    Read form input, validate it, and build the runtime config.
    """
    defaults = load_config(Path("config.json"))

    form_values = {
        "palette_source": request.form.get("palette_source", "manual"),
        "image_palette_method": request.form.get("image_palette_method", "clustered_main"),
        "image_palette_count": request.form.get("image_palette_count", "8"),
        "frame_count": request.form.get("frame_count", str(defaults["frame_count"])),
        "reconstruction_mode": request.form.get(
            "reconstruction_mode",
            str(defaults["reconstruction_mode"]),
        ),
        "score_mode": request.form.get("score_mode", str(defaults["score_mode"])),
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

    palette_source = form_values["palette_source"]
    if palette_source not in {"manual", "image"}:
        raise ValueError("Invalid palette source.")

    image_palette_method = form_values["image_palette_method"]
    if image_palette_method not in {"clustered_main", "top_frequency"}:
        raise ValueError("Invalid image-derived palette method.")

    frame_count = validate_positive_int(form_values["frame_count"], "Frame count", minimum=1)
    random_seed = validate_positive_int(form_values["random_seed"], "Random seed", minimum=0)
    gif_frame_duration_ms = validate_positive_int(
        form_values["gif_frame_duration_ms"],
        "GIF frame duration",
        minimum=1,
    )
    image_palette_count = validate_positive_int(
        form_values["image_palette_count"],
        "Image-derived palette color count",
        minimum=1,
    )

    reconstruction_mode = form_values["reconstruction_mode"]
    if reconstruction_mode not in set(get_reconstruction_mode_values()):
        raise ValueError("Invalid reconstruction mode.")

    score_mode = form_values["score_mode"]
    if score_mode not in set(get_score_mode_values()):
        raise ValueError("Invalid score mode.")

    frame_prefix = validate_safe_name(form_values["frame_prefix"], "Frame prefix")
    gif_output_name = validate_safe_name(form_values["gif_output_name"], "GIF output name")

    if not gif_output_name.lower().endswith(".gif"):
        gif_output_name = f"{gif_output_name}.gif"

    uploaded_image_path = validate_image_upload(request.files.get("source_image"))
    source_image = uploaded_image_path if uploaded_image_path else Path(defaults["source_image"])

    if not source_image.exists():
        raise ValueError("No valid source image was provided, and the default image was not found.")

    if palette_source == "manual":
        palette_colors = parse_palette_text(form_values["palette_text"])
        palette_name = form_values["selected_preset"] or "web_palette"
    else:
        source_image_path = Path(source_image)

        if image_palette_method == "clustered_main":
            palette_colors = extract_clustered_main_palette_colors(
                image_path=source_image_path,
                color_count=image_palette_count,
                preserve_darkest=True,
                preserve_lightest=True,
                min_color_distance=28.0,
                random_seed=random_seed,
            )
        else:
            palette_colors = extract_top_image_palette_colors(
                image_path=source_image_path,
                color_count=image_palette_count,
            )

        palette_name = f"image_{image_palette_method}_{len(palette_colors)}"
        form_values["palette_text"] = "\n".join(palette_colors)

    palette_size = len(palette_colors)

    config = {
        "source_image": str(source_image),
        "palette_file": str(defaults["palette_file"]),
        "palette_colors": palette_colors,
        "palette_name": palette_name,
        "frame_count": frame_count,
        "save_debug_tables": False,
        "create_gif": bool(form_values["create_gif"]),
        "gif_frame_duration_ms": gif_frame_duration_ms,
        "gif_output_name": gif_output_name,
        "frame_prefix": frame_prefix,
        "reconstruction_mode": reconstruction_mode,
        "score_mode": score_mode,
        "random_seed": random_seed,
    }

    advisory = build_frame_count_message(frame_count, palette_size)
    return config, form_values, advisory


def run_file_url(path: Path) -> str:
    """
    Convert a run file path into a browser-accessible URL.
    """
    relative_path = path.resolve().relative_to(RUNS_ROOT)
    return url_for("serve_run_file", subpath=relative_path.as_posix())


def serialize_update(update: dict) -> dict:
    """
    Convert a pipeline progress update into JSON-safe output for the UI.
    """
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
    frame_timings_seconds = list(update.get("frame_timings_seconds", []))

    percent_complete = 0
    if frame_count > 0:
        percent_complete = int((completed_frames / frame_count) * 100)

    if update.get("status") == "completed":
        percent_complete = 100

    first_frame_seconds = None
    if len(frame_timings_seconds) >= 1:
        first_frame_seconds = float(frame_timings_seconds[0])

    average_frame_seconds = None
    if len(frame_timings_seconds) >= 1:
        average_frame_seconds = float(sum(frame_timings_seconds) / len(frame_timings_seconds))

    estimated_remaining_seconds = None
    remaining_frames = max(frame_count - completed_frames, 0)

    if remaining_frames > 0:
        if average_frame_seconds is not None:
            estimated_remaining_seconds = average_frame_seconds * remaining_frames
        elif first_frame_seconds is not None:
            estimated_remaining_seconds = first_frame_seconds * remaining_frames

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
        "score_mode": update.get("score_mode"),
        "frame_urls": frame_urls,
        "gif_url": gif_url,
        "first_frame_seconds": first_frame_seconds,
        "first_frame_text": build_duration_text(first_frame_seconds) if first_frame_seconds is not None else None,
        "average_frame_seconds": average_frame_seconds,
        "average_frame_text": build_duration_text(average_frame_seconds) if average_frame_seconds is not None else None,
        "estimated_remaining_seconds": estimated_remaining_seconds,
        "estimated_remaining_text": (
            build_duration_text(estimated_remaining_seconds)
            if estimated_remaining_seconds is not None
            else None
        ),
    }


def worker(job_id: str, config: dict) -> None:
    """
    Run the pipeline in a background thread and keep job state updated.
    """
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
    """
    Render the main form page.
    """
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
        palette_preset_options=build_palette_preset_display_options(),
        reconstruction_modes=RECONSTRUCTION_MODES,
        score_mode_options=build_score_mode_options(),
    )


@app.route("/start", methods=["POST"])
def start_run():
    """
    Validate input, create a background job, and redirect to the run page.
    """
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
            "palette_source": request.form.get("palette_source", defaults["palette_source"]),
            "image_palette_method": request.form.get("image_palette_method", defaults["image_palette_method"]),
            "image_palette_count": request.form.get("image_palette_count", defaults["image_palette_count"]),
            "frame_count": request.form.get("frame_count", defaults["frame_count"]),
            "reconstruction_mode": request.form.get(
                "reconstruction_mode",
                defaults["reconstruction_mode"],
            ),
            "score_mode": request.form.get("score_mode", defaults["score_mode"]),
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
            if attempted_values["palette_source"] == "manual":
                advisory_palette_size = len(parse_palette_text(attempted_values["palette_text"]))
            else:
                advisory_palette_size = int(attempted_values["image_palette_count"])

            advisory = build_frame_count_message(
                frame_count=int(attempted_values["frame_count"]),
                palette_size=advisory_palette_size,
            )
        except Exception:
            advisory = None

        return render_template(
            "form.html",
            error=str(error),
            advisory=advisory,
            values=attempted_values,
            palette_presets=PALETTE_PRESETS,
            palette_preset_options=build_palette_preset_display_options(),
            reconstruction_modes=RECONSTRUCTION_MODES,
            score_mode_options=build_score_mode_options(),
        ), 400


@app.route("/run/<job_id>", methods=["GET"])
def run_page(job_id: str):
    """
    Render the live run progress page.
    """
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
    """
    Return live JSON progress data for a run.
    """
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
            "first_frame_seconds": None,
            "first_frame_text": None,
            "average_frame_seconds": None,
            "average_frame_text": None,
            "estimated_remaining_seconds": None,
            "estimated_remaining_text": None,
        })

    payload = serialize_update(update)
    payload["advisory"] = job.get("advisory")
    return jsonify(payload)


@app.route("/runs/<path:subpath>", methods=["GET"])
def serve_run_file(subpath: str):
    """
    Serve generated files from the run output directory.
    """
    return send_from_directory(RUNS_ROOT, subpath)


def main() -> None:
    """
    Start the Flask development server.
    """
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()