# OCR with Rust

## Status

- **Rayon Parallelized CPU Pipeline:** Implemented parallel batch processing using model cloning and gradient accumulation. Processing a batch of 64 images processes all images in parallel across all CPU cores.
- **Fast 1D Tensor Indexing:** Custom 1D vector mathematical index mapping replaces overhead-heavy bracket indexing, eliminating billions of dynamic allocations per epoch.
- **End-to-End CNN Pipeline:** Supports a LeNet-style CNN (Conv, Sigmoid/ReLU activation, Max Pooling, Flatten, Dense) with trainable parameters, momentum, learning rate scheduling/plateau decay, weight decay (L2 regularization), dropout, and image shift data augmentation.

## Run Training

```bash
cargo run --release --bin train -- \
   --train-images data/train-images-idx3-ubyte \
   --train-labels data/train-labels-idx1-ubyte \
   --num-train-images 60000 \
   --num-epochs 15 \
   --batch-size 64 \
   --learning-rate 0.02 \
   --momentum 0.9 \
   --lr-decay 0.995 \
   --val-split 0.1 \
   --lr-plateau-patience 0 \
   --lr-plateau-factor 0.5 \
   --min-learning-rate 0.00005 \
   --dropout 0.0 \
   --weight-decay 0.0 \
   --aug-shift 0 \
   --output-dir experiments
```

## Run Inference Web Server

Launch the HTTP server and access the interactive drawing canvas:

```bash
cargo run --release --bin server -- --weights-dir experiments/run_<timestamp>/weights --port 3000
```
Open your browser to `http://127.0.0.1:3000` to draw numbers and see predictions in real time.

## Recommended Defaults (Validated)

Validated with parallel mini-batch SGD on the full MNIST train set (60,000 images):

- `--num-train-images 60000`
- `--num-epochs 15`
- `--batch-size 64`
- `--learning-rate 0.02` (adjusted for batch-level weight updates)
- `--momentum 0.9`
- `--lr-decay 0.995`
- `--val-split 0.1`
- `--lr-plateau-patience 0` (disabled)
- `--min-learning-rate 0.00005`
- `--dropout 0.0`
- `--weight-decay 0.0`
- `--aug-shift 0`

Observed outcome:
- **Training Time:** Completed a full 15-epoch training session in **~35 minutes** (over **11x faster** than the previous sequential 6.7 hours run).
- **Accuracy:** Reaches **~98.5% validation accuracy** by Epoch 2.

## Quick Benchmarks

Use these presets for fast sanity checks and full-quality training.

### 1k smoke test (very fast)

Runs in **under 10 seconds** once compiled.

```bash
cargo run --release --bin train -- \
   --train-images data/train-images-idx3-ubyte \
   --train-labels data/train-labels-idx1-ubyte \
   --num-train-images 1000 \
   --num-epochs 3 \
   --batch-size 64 \
   --learning-rate 0.02 \
   --momentum 0.9 \
   --lr-decay 0.995 \
   --val-split 0.1 \
   --lr-plateau-patience 0 \
   --min-learning-rate 0.00005 \
   --dropout 0.0 \
   --weight-decay 0.0 \
   --aug-shift 0 \
   --output-dir experiments
```

### 10k dev run (medium)

Runs in **under 2 minutes**.

```bash
cargo run --release --bin train -- \
   --train-images data/train-images-idx3-ubyte \
   --train-labels data/train-labels-idx1-ubyte \
   --num-train-images 10000 \
   --num-epochs 10 \
   --batch-size 64 \
   --learning-rate 0.02 \
   --momentum 0.9 \
   --lr-decay 0.995 \
   --val-split 0.1 \
   --lr-plateau-patience 0 \
   --min-learning-rate 0.00005 \
   --dropout 0.0 \
   --weight-decay 0.0 \
   --aug-shift 0 \
   --output-dir experiments
```

### 60k full training (recommended)

Runs in **~35 minutes**.

```bash
cargo run --release --bin train -- \
   --train-images data/train-images-idx3-ubyte \
   --train-labels data/train-labels-idx1-ubyte \
   --num-train-images 60000 \
   --num-epochs 15 \
   --batch-size 64 \
   --learning-rate 0.02 \
   --momentum 0.9 \
   --lr-decay 0.995 \
   --val-split 0.1 \
   --lr-plateau-patience 0 \
   --min-learning-rate 0.00005 \
   --dropout 0.0 \
   --weight-decay 0.0 \
   --aug-shift 0 \
   --output-dir experiments
```

## CLI Parameters

- `--train-images`: path to training images IDX file (required).
- `--train-labels`: path to training labels IDX file (required).
- `--num-train-images`: number of samples to train on (optional, capped by available samples).
- `--num-epochs`: number of training epochs (default: `5`).
- `--batch-size`: mini-batch size (default: `32`).
- `--learning-rate`: base learning rate (default: `0.01`).
- `--momentum`: SGD momentum in `[0, 1)` (optional).
- `--lr-decay`: base per-epoch LR multiplier (default: `1.0`).
- `--val-split`: fraction of selected samples used for validation each epoch (default: `0.1`).
- `--lr-plateau-patience`: epochs without val-loss improvement before reducing LR (default: `0`, disabled).
- `--lr-plateau-factor`: multiplier used when LR is reduced on plateau (default: `0.5`).
- `--min-learning-rate`: lower bound for LR after decay/reduction (default: `0.00001`).
- `--dropout`: inverted dropout rate applied to dense1 and dense2 during training (default: `0.0`, disabled).
- `--weight-decay`: L2 regularization coefficient applied to all weight updates (default: `0.0`, disabled).
- `--aug-shift`: maximum random pixel shift (in each direction) applied to training images (default: `0`, disabled). `2` is a good starting value.
- `--output-dir`: base folder for saved artifacts (default: `experiments`).

Notes:

- CLI flags use kebab-case (for example, `--train-images`, not `--train_images`).
- Labels are read as single-byte MNIST label values.
- `--num-train-images` is optional and capped to available samples in the files.
- Training uses all available train samples from the provided IDX files.
- Samples are shuffled each epoch.
- A train/validation split is used each epoch and is configurable with `--val-split`.
- Momentum is optional (`--momentum 0.9` is a common choice).
- Learning-rate decay is multiplicative per epoch (`lr = lr * lr_decay`) and can be combined with validation-plateau reduction.

## Saved Results

After each training run, a folder is created under the output directory:

- `run_<timestamp>/metrics.csv`: epoch metrics (`train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`, `lr`).
- `run_<timestamp>/weights/`: final weights and biases for all trainable layers.
- `run_<timestamp>/summary.txt`: run configuration and artifact paths.

## Prepare data

Download MNIST dataset

```bash
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```

Unzip the files

```bash
gzip -d *.gz
```

## Inspect data

TRAINING SET IMAGE FILE (train-images-idx3-ubyte):

```bash
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
```

Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black). In between are grayscales.

The first 16 bytes of the file contain the magic number, the number of images, the number of rows, and the number of columns. They are all in 32-bit integers in big-endian format. The magic number for the images is 2051, and the magic number for the labels is 2049. Let's inspect the first 16 bytes of the `train-images-idx3-ubyte` file:

```bash
$ head -c 16 train-images-idx3-ubyte | od -v -A n --endian=big -t u4 --width=4
     2051
    60000
       28
       28
```

To retrieve the first image, we need to skip the first 16 bytes and read the next 28x28=784 bytes. The grayscale values are stored as unsigned bytes (0-255). Let's extract the first image:

```bash
$ head -c 800 train-images-idx3-ubyte | od -v -j 16 -A n --endian=big -t u1 --width=28
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   3  18  18  18 126 136 175  26 166 255 247 127   0   0   0   0
   0   0   0   0   0   0   0   0  30  36  94 154 170 253 253 253 253 253 225 172 253 242 195  64   0   0   0   0
   0   0   0   0   0   0   0  49 238 253 253 253 253 253 253 253 253 251  93  82  82  56  39   0   0   0   0   0
   0   0   0   0   0   0   0  18 219 253 253 253 253 253 198 182 247 241   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0  80 156 107 253 253 205  11   0  43 154   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0  14   1 154 253  90   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0 139 253 190   2   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0  11 190 253  70   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0  35 241 225 160 108   1   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0  81 240 253 253 119  25   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0  45 186 253 253 150  27   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  16  93 252 253 187   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 249 253 249  64   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0  46 130 183 253 253 207   2   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0  39 148 229 253 253 253 250 182   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0  24 114 221 253 253 253 253 201  78   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0  23  66 213 253 253 253 253 198  81   2   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0  18 171 219 253 253 253 253 195  80   9   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0  55 172 226 253 253 253 253 244 133  11   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0 136 253 253 253 212 135 132  16   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
```

or the second image:

```bash
$ head -c 1584 train-images-idx3-ubyte | od -v -j 800 -A n --endian=big -t u1 --width=28
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  51 159 253 159  50   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0  48 238 252 252 252 237   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0  54 227 253 252 239 233 252  57   6   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0  10  60 224 252 253 252 202  84 252 253 122   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0 163 252 252 252 253 252 252  96 189 253 167   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0  51 238 253 253 190 114 253 228  47  79 255 168   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0  48 238 252 252 179  12  75 121  21   0   0 253 243  50   0   0   0   0   0
   0   0   0   0   0   0   0   0  38 165 253 233 208  84   0   0   0   0   0   0 253 252 165   0   0   0   0   0
   0   0   0   0   0   0   0   7 178 252 240  71  19  28   0   0   0   0   0   0 253 252 195   0   0   0   0   0
   0   0   0   0   0   0   0  57 252 252  63   0   0   0   0   0   0   0   0   0 253 252 195   0   0   0   0   0
   0   0   0   0   0   0   0 198 253 190   0   0   0   0   0   0   0   0   0   0 255 253 196   0   0   0   0   0
   0   0   0   0   0   0  76 246 252 112   0   0   0   0   0   0   0   0   0   0 253 252 148   0   0   0   0   0
   0   0   0   0   0   0  85 252 230  25   0   0   0   0   0   0   0   0   7 135 253 186  12   0   0   0   0   0
   0   0   0   0   0   0  85 252 223   0   0   0   0   0   0   0   0   7 131 252 225  71   0   0   0   0   0   0
   0   0   0   0   0   0  85 252 145   0   0   0   0   0   0   0  48 165 252 173   0   0   0   0   0   0   0   0
   0   0   0   0   0   0  86 253 225   0   0   0   0   0   0 114 238 253 162   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0  85 252 249 146  48  29  85 178 225 253 223 167  56   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0  85 252 252 252 229 215 252 252 252 196 130   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0  28 199 252 252 253 252 252 233 145   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0  25 128 252 253 252 141  37   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
```

## CNNs

### LeNet

1. Input layer: Image 28 (width) x 28 (height) x 1 (channel)
2. Convolutional layer 1: 5x5 kernel + 2 padding: 28x28x6
3. sigmoid activation
4. Max pooling layer 1: 2x2 kernel + 2 stride: 14x14x6
5. Convolutional layer 2: 5x5 kernel + 0 padding: 10x10x16
6. sigmoid activation
7. Max pooling layer 2: 2x2 kernel + 2 stride: 5x5x16
8. flatten
9. Dense layer 1: 120 neurons
10. sigmoid activation
11. Dense layer 2: 84 neurons
12. sigmoid activation
13. Dense layer 3: 10 neurons
14. Output: 1 of 10 classes

## Containerization (Podman)

The repository supports building separate runtime images for either the training job or the inference web server from the same multi-stage `Dockerfile`.

### 1. Build Training Image
```bash
podman build --target train -t ocr-train .
```

### 2. Build Server Image
```bash
podman build --target server -t ocr-server .
```

### 3. Run Training in Container
To run training, mount the directory containing your local MNIST dataset to `/data` in the container.
*(Note: If using SELinux, append the `:Z` flag to your volume mapping)*:
```bash
podman run -v $(pwd)/data:/data:Z ocr-train \
   --train-images /data/train-images-idx3-ubyte \
   --train-labels /data/train-labels-idx1-ubyte \
   --num-train-images 60000 \
   --num-epochs 15 \
   --batch-size 64 \
   --learning-rate 0.02
```

### 4. Run Inference Server in Container
To run the server, mount your trained weights folder to `/app/weights` in the container and expose port 3000:
```bash
podman run -p 3000:3000 -v $(pwd)/experiments/run_1783808067/weights:/app/weights:Z ocr-server
```
