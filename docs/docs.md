# Neural Network — Technical Documentation

## Table of Contents

1. [Notation](#notation)
2. [Network Structure](#network-structure)
3. [Weights](#weights)
4. [Biases](#biases)
5. [Forward Pass](#forward-pass)
6. [Activation Functions](#activation-functions)
7. [Loss Function](#loss-function)
8. [Implementation Reference](#implementation-reference)

---

## Notation

| Symbol      | Meaning                                              |
|-------------|------------------------------------------------------|
| `l`         | Layer index (0 = input, 1 = first hidden, ...)       |
| `z^[l]`     | Pre-activation column vector for layer `l`           |
| `a^[l]`     | Post-activation column vector for layer `l`          |
| `W^[l]`     | Weight matrix connecting layer `l-1` to layer `l`   |
| `b^[l]`     | Bias vector for layer `l`                            |
| `g(z)`      | Activation function applied to `z`                   |

---

## Network Structure

A neural network is a series of **layers**. Each layer contains nodes (neurons). Every node in a layer connects to every node in the next layer; this is called a **fully connected** or **dense** layer.

```
Layer 0       Layer 1       Layer 2
(input)       (hidden)      (output)

  o ———————>  o  ————————>  o
  o ———————>  o
              o
```

Layers are represented as a linked list of `Layer` structs:

```
initial_layer → hidden_layer → output_layer → null
```

A `LayerIterable` is used to walk the list cleanly during the forward pass.

---

## Weights

Each node in a layer must connect to **every** node in the next layer. So the number of weights between two layers is:

```
num_weights = num_inputs × num_outputs
```

Example: a layer with 3 input nodes connecting to a layer with 5 output nodes has `3 × 5 = 15` weights.

Weights are stored as a **flattened matrix** in row-major order:

```
weights[i * num_outputs + j]  =  weight from input node i to output node j
```

### Xavier Initialization

Weights are initialized to random values in the range:

```
±1 / sqrt(num_inputs)
```

This prevents activations from exploding or vanishing as the network gets deeper. Larger input layers get smaller initial weights.

### The dot product

For a single output node `j`, its value is the dot product of the input vector and the column of weights for that node:

```
[3]   [4]
[2] · [8]  =  (3×4) + (2×8)  =  28
```

For a full layer this is a matrix multiplication: `W^[l] · a^[l-1]`

---

## Biases

A bias is a number added to a node's value **after** the weight dot product calculation.

- Biases are applied to every node **except** the input layer
- Every non-input node has its own independently learned bias
- Biases start at `0.0` and are updated during training

```
[3]   [4]          bias = 1
[2] · [8]  + 1  =  (3×4 + 2×8) + 1  =  29

[3]   [1]          bias = 4
[2] · [10] + 4  =  (3×1 + 2×10) + 4  =  27
```

---

## Forward Pass

The forward pass propagates inputs through every layer to produce a final output. It has two steps per layer.

### Step 1: Pre-activation (z)

Matrix multiply the weight matrix by the activations from the previous layer, then add the bias vector:

```
z^[l] = W^[l] · a^[l-1] + b^[l]
```

In code, for each output node `j`:

```
z[j] = bias[j] + sum over i of (inputs[i] * weights[i * num_outputs + j])
```

`z^[l]` is the **unprocessed** output of the raw weighted sums before any activation.

### Step 2: Post-activation (a)

Pass each `z` value through the activation function `g`:

```
a^[l] = g(z^[l])
```

`a^[l]` is the **final** output of the layer, and becomes the input `a^[l-1]` for the next layer.

### Full chain

```
inputs (a^[0])
   ↓
z^[1] = W^[1] · a^[0] + b^[1]
a^[1] = g(z^[1])
   ↓
z^[2] = W^[2] · a^[1] + b^[2]
a^[2] = g(z^[2])
   ↓
... and so on until the output layer
```

---

## Activation Functions

After computing `z`, we need to **squash** the value into a usable range. This is what activation functions do.

An activation function must be:
1. **Non-linear** — cannot be in the form `g(z) = mz + b`. Without this, the entire network collapses to a simple linear function no matter how many layers it has.
2. **Bounded** — compresses input to a fixed range like `0 to 1`.

### Sigmoid

```
g(z) = 1 / (1 + e^(-z))
```

Output range: `(0, 1)`

Sigmoid models are how a real neuron fires, it is either off (near 0), or on (near 1), or somewhere in the transition window. As `z → +∞`, `g(z) → 1`. As `z → -∞`, `g(z) → 0`.

```
z = -5  →  g(z) ≈ 0.007
z =  0  →  g(z) = 0.500
z =  5  →  g(z) ≈ 0.993
z = 27  →  g(z) ≈ 0.999
```

Implementation note: clamp `z` to `(-500, 500)` before computing `exp` to prevent float overflow.

### ReLU (for later)

```
g(z) = max(0, z)
```

More common in modern networks for hidden layers. Not used here yet.

---

## Loss Function

The loss function measures **how wrong the network's prediction is**. The goal of training is to minimize this number.

### Mean Squared Error (MSE)

```
loss = (output - target)^2
```

For multiple outputs:

```
loss = (1/n) × sum((output[i] - target[i])^2)
```

A loss of `0` means perfect prediction. A loss of `0.297` (as seen in testing) means the output is off by about `0.545` from the target.

---

## Implementation Reference

### Layer struct

```
const Layer -> struct {
    num_inputs:  int,
    num_outputs: int,
    weights:     *float,   // size: num_inputs * num_outputs
    biases:      *float,   // size: num_outputs
    z:           *float,   // pre-activation, size: num_outputs
    outputs:     *float,   // post-activation (a), size: num_outputs
    next:        *Layer,
};
```

### Key functions

| Function | Description |
|---|---|
| `init_layer(in, out)` | Allocates a layer with given dimensions |
| `init_weights_biases()` | Xavier weight init, biases to 0 |
| `forward(inputs)` | Computes z and a for this layer |
| `print_forward()` | Prints z and a vectors |
| `free_layer(l)` | Frees all heap memory recursively |
| `iter_init(l)` | Creates an iterator over the layer linked list |
| `mse_loss(outputs, targets, n)` | Computes mean squared error |

### Memory layout

All weight/bias/activation arrays are heap-allocated via `alloc` and must be freed with `free_layer`. The `next` pointer links layers into a singly linked list. Stack-allocated layers must not be freed via `free_layer`'s recursive `free(l.next)` path only heap-allocated next pointers should be freed.

---

## What's Next

The network can currently **predict** but not **learn**. Learning requires:

1. **Backpropagation** — compute how much each weight contributed to the loss by working backwards through the chain rule
2. **Gradient descent** — nudge each weight in the direction that reduces loss
3. **Training loop** — repeat forward pass + backprop over many examples until loss converges