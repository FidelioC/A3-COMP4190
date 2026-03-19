# How to run

## Setup

Install dependencies (if needed):

```bash
pip install -r requirements.txt
```

A sample noisy image (`noisy_image.txt`) is already provided in the project directory.

## Two different ways

### Command line

Run `Problem1.py` with the path to a noisy image text file as the argument:

```bash
python Problem1.py noisy_image.txt
```

This will display a side-by-side plot of the noisy input and the denoised output, and save the result to `problem1_output.png`.

### Jupyter notebook

Open and run `Problem1.ipynb`:

```bash
jupyter lab Problem1.ipynb
```

> NOTE: Initial setup of Jupyter kernel may be needed.
