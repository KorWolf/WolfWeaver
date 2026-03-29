# Tapestry Pipeline

A local web-based tool for transforming images into limited color palettes while strictly controlling how many pixels each color is used.

Instead of just “approximating” colors, this pipeline enforces exact color quotas and lets you explore different ways those colors are arranged across the image.

The long term goal is to get this to a point where I can make detailed weaving patterns form any image.
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
      Note: creating a palette from your image has a varity of additional options.
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
   * config of settings used
7. Next run:

   * from current settings and image
   * from default

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

## Score modes

Score modes control how source colors are matched to palette colors.

This has a big impact on how colors shift during reconstruction.

### Basic / Weighted RGB

* Uses RGB difference with some light weighting
* Fast and predictable

### Accent aware

* Protects strong colors (eyes, highlights, accents)
* Helps prevent vivid details from being washed out

### Separation aware

* Pushes similar colors apart more aggressively
* Helps avoid color blending (especially skin/hair/clothing overlap)

### Perceptual Lab

* Uses perceptual color distance instead of raw RGB
* Often produces more natural-looking matches

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
Note: match frame count to total colors for max color rotation frames, they make fun GIFs

### Image-derived palettes

Instead of entering colors manually, you can also generate a palette directly from the source image.

Options include:

* clustered main colors (various selection strategies)
* top frequency (most common exact colors)

You can also control how many colors are extracted.

Different methods will favor:

* accuracy (closest real colors)
* frequency (most used colors)
* or diversity (spread across color families)

This is where a lot of the “look” of the final output comes from.

---

## Post-processing

Post-processing runs after the main reconstruction step.

These modes try to help clean up noise and improve visual cohesion (not well calibrated yet).

### None

* No changes after reconstruction

### Coherence (basic)

* Smooths isolated pixels
* Improves region consistency

### Coherence (edge aware)

* Smooths noise while preserving strong edges
* Helps keep shapes and outlines clearer

---

## Performance notes

* Larger palettes increase processing time per frame
* More frames increase total runtime
* Larger images increase total runtime as this breaks down the images by the pixel
   timy images ~ a few seconds to generate each frame even on more complex settings
* This project prioritizes **correctness and clarity over speed**

---

## Known behaviors / quirks
* Images with transparency (alpha):
   * Presently these 'break' the math in interesting ways
      * Using reconstruction mode Scanline causes banding/striping
      * Using reconstruction mode Weighted_Random causes a dithering/static look

---

## Support & Usage
This project is free for personal use.

If you find it useful and want to support development:
Buy me a coffee: https://ko-fi.com/korwolf

---

## Comercial Use

If you wish to use this for:
* business
* client work
* or any revenue-generating project
please reach out for licensing.

---

## Want me to process images for you?

If you don't want to run the pipeline yourself, I can generate images and GIFs for you.
* 1$ minimun per image run
* Final cost depends on time and image complexity
Requests can be made through my Ko-fi (https://ko-fi.com/korwolf).

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
