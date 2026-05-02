# neural_net

A feedforward neural network built from scratch in [Luma](https://luma-website-mu.vercel.app/), compiled to native code via LLVM. No ML libraries — just weights, matrix math, and sigmoid.

## Architecture

```
inputs → [ hidden layers ] → output
  2    →   5  →  5        →   1
```

Each layer is fully connected to the next. Every connection has a weight, every non-input node has a bias.

## Project Structure

```
src/
  main.lx       — entry point, network setup and forward pass
  layer.lx      — Layer struct, LayerIterable, forward pass, weight init
  helper.lx     — random_float utility
std/
  libc.lx       — extern bindings to libc (stdio, stdlib, math)
bin/
  neural        — compiled output
lumix.toml      — build config
```

## Building

```bash
lumix build
./bin/neural
```

Or manually:

```bash
luma src/main.lx -l std/libc.lx src/helper.lx src/layer.lx -O3 --no-sanitize -name bin/neural
```

## How It Works

### Layers

A `Layer` holds its weights, biases, and the results of the forward pass:

```
const Layer -> struct {
    num_inputs:  int,
    num_outputs: int,
    weights:     *float,   // flattened matrix, size = num_inputs * num_outputs
    biases:      *float,   // size = num_outputs
    z:           *float,   // pre-activation values
    outputs:     *float,   // post-activation values (a)
    next:        *Layer,   // linked list to next layer
};
```

Layers are linked together as a singly linked list and traversed with `LayerIterable`.

### Weight Initialization

Weights are initialized using **Xavier initialization** — random values in the range `±1/sqrt(num_inputs)`. This keeps activations from exploding or vanishing in deeper networks.

Biases start at `0.0`.

### Forward Pass

For each layer, two steps:

**Step 1 — compute z (pre-activation):**
```
z[j] = bias[j] + sum(inputs[i] * weights[i * num_outputs + j])
```

**Step 2 — compute a (post-activation):**
```
a[j] = sigmoid(z[j]) = 1 / (1 + e^(-z[j]))
```

The output `a` of each layer becomes the input to the next layer.

### Loss

Mean squared error measures how wrong the network is:

```
loss = (output - target)^2
```

A loss of `0` means perfect prediction. Training drives this number toward zero.

## Example Output

```
-- Layer 1 --
  z = [ 0.070509, 0.008186, 0.544103, -1.052420, 0.557907 ]
  a = [ 0.517620, 0.502047, 0.632766, 0.258761, 0.635968 ]
-- Layer 2 --
  z = [ -0.077389, -0.154753, 0.407702, ... ]
  a = [ 0.480662, 0.461389, 0.600537, ... ]
-- Layer 3 --
  z = [ -0.181528 ]
  a = [ 0.454742 ]
Loss: 0.297306
```

## Status

- [x] Layer struct with linked list
- [x] Xavier weight initialization  
- [x] Forward pass (z and a)
- [x] Sigmoid activation
- [x] MSE loss
- [x] Backpropagation
- [x] Weight updates (gradient descent)
- [x] Training loop