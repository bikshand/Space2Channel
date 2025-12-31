import tensorflow as tf
import numpy as np

# -----------------------------
# Parameters
# -----------------------------
B, H, W = 1, 32, 64     # batch, height, width
K = 5                  # kernel size along H
F = 8                  # width folding factor
Cout = 1               # output channels

assert W % F == 0

# -----------------------------
# Input tensor (NHWC)
# -----------------------------
x = tf.random.normal((B, H, W, 1))

# -----------------------------
# Original filter + bias
# -----------------------------
filterVal = tf.random.normal((K, 1, 1, Cout))
biasVal = tf.random.normal((Cout,))

# -----------------------------
# Width folding: W -> W/F, Cin -> F
# -----------------------------
x_folded = tf.reshape(x, (B, H, W // F, F))

# -----------------------------
# Build diagonal folded filter
# -----------------------------
filterValNew = np.zeros((K, 1, F, F * Cout), dtype=np.float32)
for f in range(F):
    filterValNew[:, :, f, f*Cout:(f+1)*Cout] = np.squeeze(filterVal.numpy(), axis=-1)

filterValNew = tf.constant(filterValNew)
biasValNew = tf.tile(biasVal, [F])

# =====================================================
# Timed kernels
# =====================================================

@tf.function
def conv_original(x, w, b):
    y = tf.nn.conv2d(
        x, w,
        strides=[1, 1, 1, 1],
        padding="VALID",
        data_format="NHWC"
    )
    return tf.nn.bias_add(y, b)

@tf.function
def conv_folded(x, w, b):
    y = tf.nn.conv2d(
        x, w,
        strides=[1, 1, 1, 1],
        padding="VALID",
        data_format="NHWC"
    )
    return tf.nn.bias_add(y, b)

# -----------------------------
# Warm-up (important)
# -----------------------------
for _ in range(10):
    _ = conv_original(x, filterVal, biasVal)
    _ = conv_folded(x_folded, filterValNew, biasValNew)


# -----------------------------
# Timing
# -----------------------------
num_iters = 100

# Original convolution timing
start = tf.timestamp()
for _ in range(num_iters):
    y_orig = conv_original(x, filterVal, biasVal)
_ = y_orig.numpy()  # <-- GPU sync barrier
end = tf.timestamp()

orig_time_ms = (end - start) * 1000 / num_iters

# Folded convolution timing
start = tf.timestamp()
for _ in range(num_iters):
    y_folded = conv_folded(x_folded, filterValNew, biasValNew)
_ = y_folded.numpy()  # <-- GPU sync barrier
end = tf.timestamp()

folded_time_ms = (end - start) * 1000 / num_iters

# -----------------------------
# Reconstruct original layout
# -----------------------------
y_reconstructed = tf.reshape(
    y_folded,
    (B, y_folded.shape[1], W)
)

# -----------------------------
# Verification
# -----------------------------
max_error = tf.reduce_max(
    tf.abs(tf.squeeze(y_orig, axis=-1) - y_reconstructed)
)

print(f"Original conv: {orig_time_ms.numpy():.4f} ms")
print(f"Folded conv:   {folded_time_ms.numpy():.4f} ms")
print("Speedup:", orig_time_ms.numpy() / folded_time_ms.numpy())
print("Max absolute error:", max_error.numpy())

tf.debugging.assert_near(
    tf.squeeze(y_orig, axis=-1),
    y_reconstructed,
    atol=1e-5
)

print("✔ Width folding transformation is numerically correct")

