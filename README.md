# Tapestry Pipeline

A local web-based tool for transforming images into limited color palettes while strictly controlling how many pixels each color is used.

Instead of just “approximating” colors, this pipeline enforces exact color quotas and lets you explore different ways those colors are arranged across the image.

---

## What it does

* Loads an image (uploaded or default)
* Breaks it into all unique source colors
* Compares those colors to a smaller replacement palette
* Assigns each pixel to a replacement color using strict quotas
* Rebuilds the image using different reconstruction methods
* Generates:

  * individual frames
  * optional animated GIFs

---

## Why this is different

Most palette reduction tools focus on “closest match.”

This project instead focuses on:

* **Exact color counts** (every palette color is used a specific number of times)
* **Multiple reconstruction styles** (how colors are distributed matters)
* **Repeatable vs exploratory outputs**
* **Transparency of process** (CSV outputs, visible steps)

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/your-username/tapestry-pipeline.git
cd tapestry-pipeline
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate
```

**Mac/Linux:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the app

```bash
python -m src.flask_app
```

Then open:

```text
http://127.0.0.1:5000
```

---

## How to use

1. Upload an image (or use default)
2. Choose a palette preset or enter hex values manually
3. Set:

   * frame count
   * reconstruction mode
   * optional GIF output
4. Start run
5. Watch:

   * live progress
   * timer
   * frames as they generate
6. Download:

   * frames
   * GIF

---

## Reconstruction modes

Each mode changes how colors are distributed across the image:

### Scanline

* Deterministic
* Processes pixels top-left → bottom-right
* Stable and structured output

### Random (seeded)

* Random placement, but repeatable
* Same seed = same result

### Random (unseeded)

* Fully random each run
* Good for experimentation

### Weighted random

* Random placement with bias toward closer color matches
* Produces more natural-looking results while preserving variation

---

## Palette tools

* Manual hex palette input
* Built-in presets:

  * monochrome sets
  * complementary pairs
  * contrast palettes
  * retro / themed palettes
* Live color swatch preview
* Editable after selection

---

## Performance notes

* Larger palettes increase processing time per frame
* More frames increase total runtime
* This project prioritizes **correctness and clarity over speed**

---

## Tech stack

* Python
* Flask
* NumPy
* Pillow
* Pandas
* ImageIO

---

## Inspiration

This project was inspired by the Tumblr post by nosferatuprivilege:
https://www.tumblr.com/teawitch/812178955121623040?source=share


---

## Future ideas

* Additional reconstruction modes (block-based, dithering, etc.)
* Palette grouping and filtering
* Performance improvements
* Export/import configs

---

