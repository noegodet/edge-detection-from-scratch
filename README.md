# Edge detection from scratch

Gaussian filtering, convolution and edge detection written by hand in Python, without calling OpenCV's filtering functions. The experiments show how filter size and image scale change what an edge detector sees, and compare a Gaussian derivative detector with the Gb boundary detector on colour images.

Course homework, Computer Vision, Politehnica University of Bucharest (Erasmus+ semester, October 2026).

## What is implemented

All the image processing below is written by hand. OpenCV is used only to read and resize images, and in the test as a reference.

| Module | Content |
|---|---|
| `edges/kernels.py` | 2D Gaussian kernel, derivative of Gaussian along each axis, Gb step kernel |
| `edges/convolution.py` | 2D convolution with zero padding, compiled with Numba |
| `edges/detectors.py` | Grey-level gradient magnitude; colour boundary strength from the 2x2 structure tensor summed over the R, G, B layers (square root of its largest eigenvalue); GD and Gb detectors on RGB |
| `run_experiments.py` | Runs every experiment and saves the figures as PNG |
| `tests/test_convolution.py` | Checks the convolution against `cv2.filter2D` |

The hand-written convolution matches `cv2.filter2D` to within 1e-13 on every kernel tested.

## Experiments

Each experiment runs on every input image, which is first resized to a width of at most 800 px.

- **Experiment 1, bigger filter on the same image:** filters of 3x3, 7x7 and 15x15, with sigma = size / 6.
- **Experiment 2, same filter on a smaller image:** a 7x7 filter on the image at scale 1, 1/2 and 1/4.

**Part A** applies the Gaussian derivative (GD) detector to the grey image. **Part B** applies GD and Gb to the three colour layers. For Gb, sigma is half the window radius and the step parameter is 1 pixel.

## Results

### Part A: grey-level Gaussian derivative

![Bigger filter, same image](docs/part_a_filter_size.png)

**Filter size.** A small filter gives thin, precise edges but many spurious responses from texture and noise. As the filter grows, the spurious responses decrease and the edges get thicker. On a photo of Brașov, the roof tiles are visible at 3x3 but disappear at 15x15, while the outlines of the buildings remain.

![Same filter, smaller image](docs/part_a_image_scale.png)

**Image scale.** Sigma controls how much smoothing is applied, so it sets the size of the structures the detector can see. With a fixed filter, shrinking the image by a factor k is equivalent to multiplying sigma by k. This is why experiment 2 reaches the same conclusion as experiment 1: the edges get thicker as the image gets smaller. On a dense city scene, the image becomes hard to read as the edges thicken.

### Part B: colour detectors, GD and Gb

![GD vs Gb with a 15x15 filter](docs/part_b_gd_vs_gb.png)

Both colour detectors behave like the grey-level detector of Part A. At a small scale (3x3 filter or full-size image), they detect many thin edges, including textures and noise. At a large scale (15x15 filter or image reduced to 1/4), only the main object boundaries remain.

**Gb keeps thinner edges than GD at large scales.** Its kernel models a sharp step, so the response stays concentrated near the true boundary even when the window is large.

## Run it

```bash
pip install -r requirements.txt
python run_experiments.py --images images --out outputs
python -m tests.test_convolution
```

Put your own images in `images/`. The script writes 6 figures per image into `outputs/part_a/` and `outputs/part_b/`. `--parts a` or `--parts b` runs a single part.

## Notes

- **Images are not included.** The course images came from the web and are not mine to publish. The figures in `docs/` use my own photos.
- **Convolution or correlation.** `correlate2d` slides the kernel without flipping it; `convolve2d` flips it first, which is a true convolution. For the symmetric Gaussian kernel the two give the same result. For derivative kernels only the sign changes, so the edge magnitudes are identical.
- **Borders.** Zero padding makes the image border appear as an edge, visible as a thin frame on the edge maps.

## Reference

Gb detector: M. Leordeanu, R. Sukthankar, C. Sminchisescu, *Generalized Boundaries from Multiple Image Interpretations*, IEEE TPAMI, 2014.
