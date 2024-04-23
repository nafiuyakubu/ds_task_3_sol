# This project is a toy project for training and quality assurance purposes

Task 3
We want to test your research ability.

We have 20 pictures of motherboards with memory.
We have 20 pictures of motherboards without memory.

We want to make a simple flask api where we upload image or choose a hardcode image and can draw a box around the different memory. So if there 2 memory, we draw 2 boxes. If only 1 memory, draw 1 box only.

Which algorithm you will use? Why you choose this? Does CPU or GPU have an impact on your decision?

What if given a video instead of images? Does your approach change?

We suggest using https://www.makesense.ai/ to annotate.

# ------------------------------------------

# ------------------------------------------

# ------------------------------------------

Algorithm used is a Single-Shot Object Detection (SSD) model.

1 [Efficiency: SSD models are known for being relatively fast and lightweight
compared to other object detection models, making them suitable for real-time
applications like this.]

2 [Accuracy: SSD models can achieve good accuracy for object detection tasks,
especially for well-defined objects like memory modules on a motherboard.]

3 [Simplicity: SSD models are relatively simpler to implement compared to some other
advanced detection models.]

Why using the MobileNet-SSD Model
MobileNet-SSD is a lightweight variant of SSD that uses MobileNet as its backbone network.
It offers a good balance between accuracy and speed, making it suitable for real-time applications and resource-constrained environments.
MobileNet-SSD models can be deployed efficiently in a Flask API, providing fast inference without compromising too much on accuracy.

CPU vs GPU Impact
GPU Preferred: While the project can technically run on a CPU,
using a GPU will significantly improve processing speed,
especially for real-time video processing.
GPUs are specifically designed for handling the parallel computations
involved in deep learning tasks like object detection.

If your model is small and the inference time is low, using CPU should suffice.
Flask APIs typically handle low to moderate traffic well with CPU.
If your model is computationally intensive and the inference time is high,
using GPU can significantly speed up the inference process,
especially when handling multiple requests simultaneously.
This can be beneficial for scaling up your API to handle higher traffic.
However, deploying GPU-based models often requires additional setup
and cost compared to CPU-based deployment.

Video Processing Approach
My approach can be adapted for video processing with some modifications:

Frame-by-Frame processing: The video will be broken down into individual frames.
The SSD model will be applied to each frame to detect memory modules.

Object Tracking (Optional): For a more robust solution, you can implement object
tracking techniques to track the detected memory modules across frames.
This can help handle slight movements or camera shakes.

STEPS IN BUILDING MY CUSTOM MODEL

(1)Create a Training dataset
(2)Train a model with custom dataset[Transfer learning]
(3)Deploy a model to the App

When to use each annotation type:
Use Rectangles When:
Objects in your dataset are primarily rectangular or have simple shapes.
Speed and efficiency in annotation are critical.
Your object detection algorithm can work well with bounding box annotations.
Use Polygons When:
Objects in your dataset have irregular shapes or complex boundaries.
Precise object localization is crucial, especially for tasks like instance segmentation.
You need to reduce background noise and accurately capture object contours.
