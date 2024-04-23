from flask import Blueprint, request, render_template, jsonify, abort, make_response, Response
from PIL import Image, ImageDraw
import numpy as np
# import tensorflow as tf
# from tf.keras.applications.efficientnet import preprocess_input  # Import the preprocessing function specific to EfficientNet
import io
import base64

detection_bp = Blueprint('detection', __name__)

# Load your trained model
model = tf.keras.models.load_model('models/custom_model.tflite')
# Define class labels (if available)
class_labels = ['Memory']


# INPUT_SIZE = (224, 224)  # Example input size for EfficientNet
# Function to preprocess image
def preprocess_image(image):
    # Resize the image to match the input size of your model
    image = image.resize((224, 224))  # Example input size for EfficientNet
    # Convert image to numpy array
    image_array = np.array(image)
    # Preprocess the image according to the preprocessing function of your model
    processed_image = tf.keras.applications.efficientnet.preprocess_input(image_array)
    
    return processed_image


# Function to draw bounding boxes
def draw_boxes(image, boxes, class_names):
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(boxes):
        ymin, xmin, ymax, xmax = box
        left = xmin * image.width
        top = ymin * image.height
        right = xmax * image.width
        bottom = ymax * image.height
        class_name = class_names[i] if class_names else None
        label = class_name if class_name else f"Object {i+1}"
        draw.rectangle([left, top, right, bottom], outline="red", width=2)
        draw.text((left, top), label, fill="red")
    return image
# def draw_boxes(image, boxes, class_names):
#     draw = ImageDraw.Draw(image)
#     for box in boxes:
#         ymin, xmin, ymax, xmax = box
#         left = xmin * image.width
#         top = ymin * image.height
#         right = xmax * image.width
#         bottom = ymax * image.height
#         draw.rectangle([left, top, right, bottom], outline="red", width=2)
#     return image


@detection_bp.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    image_file = request.files['image']
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Make predictions
    predictions = model.predict(np.expand_dims(processed_image, axis=0))
    # Process predictions and draw bounding boxes
    predictions = model.predict(np.expand_dims(processed_image, axis=0))
    # Extract bounding box coordinates and class labels from predictions
    boxes = predictions[:, :, :4]  # Assuming predictions are in the format (batch_size, num_boxes, 4) for box coordinates
    class_indices = np.argmax(predictions[:, :, 4:], axis=-1)  # Assuming predictions are in the format (batch_size, num_boxes, num_classes)
    # Convert class indices to class labels
    class_labels = [class_labels[idx] for idx in class_indices]
    # Draw bounding boxes on the image
    image_with_boxes = draw_boxes(image, boxes, class_labels)

    # Convert image to base64 string
    buffered = io.BytesIO()
    image_with_boxes.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return render_template('index.html', result_image=img_str)