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

## Reusing settings

After a run completes, you can click “Use these settings” to load the same configuration back into the form.

This lets you:

* tweak a single setting
* rerun quickly
* iterate without re-entering everything

If no new image is uploaded, the previous image will be reused automatically.

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

### Checker (seeded)

* Places colors in a checker-like distribution pattern
* Still influenced by randomness, but with structure
* Can produce more evenly spread results than pure random

### Block (seeded)

* Groups pixels into local regions (“blocks”)
* Colors tend to stay clustered instead of scattered
* Produces more painterly or patch-like results

### Weighted random

* Random placement with bias toward closer color matches
* Produces more natural-looking results while preserving variation

---

## Score modes

Score modes control how source colors are matched to palette colors.

This has a big impact on how colors shift during reconstruction.

### Basic RGB / S / L

* Baseline scoring mode
* Compares RGB values along with saturation and lightness
* Good general-purpose starting point

### Weighted RGB / S / L

* Similar to the basic mode, but with stronger weighting
* Gives more influence to some differences than the baseline mode
* Useful when the basic mode feels a little too loose

### Accent aware

* Helps protect stronger accent colors
* Useful for things like eyes, highlights, small colored details, or stylized features
* Can help keep vivid details from getting washed out

### Separation aware

* Pushes similar colors apart more aggressively
* Helps avoid color blending (especially skin/hair/clothing overlap)
* Can help with images where skin, hair, fabric, or background colors start collapsing into each other

### Perceptual Lab

* Uses perceptual color distance instead of raw RGB
* Often produces more natural-looking matches
* A strong choice for more realistic or painterly images

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

## Run experience

While a run is processing, the UI shows:

* current stage of the pipeline
* frame progress
* runtime timer
* estimated remaining time (after first frame)

The pipeline is intentionally transparent so you can see what it is doing rather than just waiting on a blank screen.

---

## Tips

The UI includes a quick tips section with suggested settings for different image types.

More detailed examples will be added over time and a deep dive page is planned for the future.

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
* Import configs/save settings as preset
* Ability to generate the image inside a grid for weaving patterns
* Hex code matcher to common thread/yarn colors/names
* Deep dive into settings and their outputs

---
