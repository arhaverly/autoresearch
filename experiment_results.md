# Experiment Results



Explanation: Explanation of config updates

To improve MNIST results while still keeping training quick, I’d make a few targeted changes:

1. Switch from `mlp` to `cnn`
   - MNIST is image data, and a CNN generally performs much better than an MLP on this task.
   - This is usually the biggest accuracy improvement available from the current config options.

2. Increase CNN capacity slightly
   - Change `cnn_channels` from `[32, 64]` to `[32, 64, 128]`.
   - This gives the model better feature extraction while remaining lightweight enough for a short run.

3. Train a bit longer
   - Increase epochs from `3` to `5`.
   - Three epochs is often too short to let a CNN settle; five is still fast on MNIST.

4. Increase batch size
   - Change `batch_size` from `64` to `128`.
   - This speeds up training and is usually safe for MNIST.

5. Use a slightly higher learning rate for Adam
   - Increase `learning_rate` from `0.001` to `0.0015`.
   - CNNs on MNIST often tolerate this well and converge faster in short runs.

6. Reduce dropout a bit
   - Lower dropout from `0.2` to `0.1`.
   - For MNIST with a modest CNN and short training, too much dropout can slow learning.

7. Make scheduler actually matter during this short run
   - Current `step_size` is `6`, but training only runs a few epochs, so it never activates.
   - Set `step_size` to `2` so LR decays during the run.

8. Slightly reduce augmentation strength
   - Lower `random_rotation` from `8` to `5`.
   - Mild augmentation helps, but too much can make MNIST digit classification harder in short training.

Accuracy: 0.993
