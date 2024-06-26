# Imports
import numpy as np
import os

from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

from tflite_support import metadata

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

# Confirm TF Version
print("\nTensorflow Version:")
print(tf.__version__)
print()


# Load Dataset
train_data = object_detector.DataLoader.from_pascal_voc(
    'custom_data/train',
    'custom_data/train',
    ['memory']
)

val_data = object_detector.DataLoader.from_pascal_voc(
    'custom_data/validate',
    'custom_data/validate',
    ['memory']
)


# spec = model_spec.get('efficientdet_lite0')
# model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=20, validation_data=val_data)

# Load model spec
spec = object_detector.EfficientDetSpec(
  model_name='efficientdet-lite2',
  uri='https://tfhub.dev/tensorflow/efficientdet/lite2/feature-vector/1',
  model_dir='/content/checkpoints',
  hparams={'max_instances_per_image': 8000})

# Train the model
model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=20, validation_data=val_data)

# Evaluate the model
eval_result = model.evaluate(val_data)

# Print COCO metrics
print("COCO metrics:")
for label, metric_value in eval_result.items():
    print(f"{label}: {metric_value}")

# Add a line break after all the items have been printed
print()

# Export the model
model.export(export_dir='.', tflite_filename='custom_model.tflite')

# Evaluate the tflite model
tflite_eval_result = model.evaluate_tflite('custom_model.tflite', val_data)

# Print COCO metrics for tflite
print("COCO metrics tflite")
for label, metric_value in tflite_eval_result.items():
    print(f"{label}: {metric_value}")