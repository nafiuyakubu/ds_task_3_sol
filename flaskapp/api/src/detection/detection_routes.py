from flask import Blueprint, request, render_template, jsonify, abort, make_response, Response
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import io
import base64

detection_bp = Blueprint('detection', __name__)

# Load your trained model
interpreter = tf.lite.Interpreter(model_path="models/custom_model.tflite")
interpreter.allocate_tensors()
# # Define class labels (if available)
class_labels = ['Memory']

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@detection_bp.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    # image_file = request.files['image']
    # image_bytes = image_file.read()
    # image = Image.open(io.BytesIO(image_bytes))
    image = Image.open(request.files['image'])
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)

    # Check if the input image shape matches the model input shape
    if image.shape != tuple(input_details[0]['shape']):
        return jsonify({'error': 'Input image shape does not match model input shape'})

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image.astype(np.float32))

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Post-process predictions (e.g., filter by confidence threshold)
    detections = [(CLASSES[i], float(output_data[0][i])) for i in range(len(output_data[0]))]

    return render_template('index.html', result_image=detections)
    # return jsonify({'detections': detections})

    # # Preprocess the image
    # processed_image = preprocess_image(image)
    # # Make predictions
    # predictions = model.predict(np.expand_dims(processed_image, axis=0))
    # # Process predictions and draw bounding boxes
    # predictions = model.predict(np.expand_dims(processed_image, axis=0))
    # # Extract bounding box coordinates and class labels from predictions
    # boxes = predictions[:, :, :4]  # Assuming predictions are in the format (batch_size, num_boxes, 4) for box coordinates
    # class_indices = np.argmax(predictions[:, :, 4:], axis=-1)  # Assuming predictions are in the format (batch_size, num_boxes, num_classes)
    # # Convert class indices to class labels
    # class_labels = [class_labels[idx] for idx in class_indices]
    # # Draw bounding boxes on the image
    # image_with_boxes = draw_boxes(image, boxes, class_labels)

    # # Convert image to base64 string
    # buffered = io.BytesIO()
    # image_with_boxes.save(buffered, format="PNG")
    # img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # return render_template('index.html', result_image=img_str)