# RadarLocNet

This repository provides pytorch implementations of [Under the Radar](https://arxiv.org/abs/2001.10789) and our own learning-based radar odometry network RadarLocNet.

"Under the Radar" uses a neural network that learn to extract keypoints directly from radar data. They use differentiable point matching and a differentiable SVD to estimate rotation and translation directly. The error with respect to the ground truth is then backpropagated through the network.

We trained and tested these networks on the [Oxford Radar Robotcar Dataset](https://oxford-robotics-institute.github.io/radar-robotcar-dataset/).
