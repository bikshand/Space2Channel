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
# Conv along H only -> kernel (K,1)
# -----------------------------
filterVal = tf.random.normal((K, 1, 1, Cout))
biasVal = tf.random.normal((Cout,))

# -----------------------------
# Original convolution
# -----------------------------
y_orig = tf.nn.conv2d(
    x,
    filterVal,
    strides=[1, 1, 1, 1],
    padding="VALID",
    data_format="NHWC"
)
y_orig = tf.nn.bias_add(y_orig, biasVal)

# -----------------------------
# Width folding: W -> W/F, Cin -> F
# -----------------------------
# (B, H, W, 1) -> (B, H, W/F, F)
x_folded = tf.reshape(x, (B, H, W // F, F))

# -----------------------------
# Build diagonal filter
# (K,1,1,Cout) -> (K,1,F,F*Cout)
# -----------------------------
filterValNew = np.zeros((K, 1, F, F * Cout), dtype=np.float32)

for f in range(F):
    filterValNew[:, :, f, f*Cout:(f+1)*Cout] = np.squeeze(filterVal.numpy(),axis=-1)

filterValNew = tf.constant(filterValNew)

# -----------------------------
# Bias replication
# -----------------------------
biasValNew = tf.tile(biasVal, [F])

# -----------------------------
# Folded convolution
# -----------------------------
y_folded = tf.nn.conv2d(
    x_folded,
    filterValNew,
    strides=[1, 1, 1, 1],   # stride only along H
    padding="VALID",
    data_format="NHWC"
)
y_folded = tf.nn.bias_add(y_folded, biasValNew)

# -----------------------------
# Reconstruct original layout
# (B, H', W/F, F) -> (B, H', W)
# -----------------------------
y_reconstructed = tf.reshape(
    y_folded,
    (B, y_folded.shape[1], W)
)

# -----------------------------
# Verification
# -----------------------------
max_error = tf.reduce_max(tf.abs(np.squeeze(y_orig, axis=-1) - y_reconstructed))
print("Max absolute error:", max_error.numpy())

# Assert correctness
tf.debugging.assert_near(np.squeeze(y_orig, axis=-1), y_reconstructed, atol=1e-5)
print("✔ Width folding transformation is numerically correct")

