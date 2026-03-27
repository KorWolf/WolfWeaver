# Tapestry Pipeline

A Python-based image transformation pipeline for remapping images into a constrained replacement palette with exact per-color quotas.

## Goals

- Load an image
- Analyze unique colors and frequencies
- Compare source colors to a replacement palette
- Assign replacement colors using a quota-constrained method
- Rebuild the final image
- Generate frame sequences and GIFs

## Tech stack

- Python
- NumPy
- Pillow
- Pandas
- ImageIO

## Inspiration

This project was inspired by the tumbler post by nosferatuprivilege located here: https://www.tumblr.com/teawitch/812178955121623040?source=share
