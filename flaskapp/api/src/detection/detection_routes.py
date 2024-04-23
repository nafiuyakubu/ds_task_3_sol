from flask import Blueprint, request, render_template, jsonify, abort, make_response, Response
from PIL import Image, ImageDraw
import cv2
import numpy as np
import tensorflow as tf
import io
import base64
import binascii  # Import binascii module for base64 encoding


detection_bp = Blueprint('detection', __name__)

model_uri = "models/custom_model.tflite"
# Load your trained model
interpreter = tf.lite.Interpreter(model_path=model_uri)
interpreter.allocate_tensors()
# # Define class labels (if available)
class_labels = ['Memory']

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# input details
# print(input_details)
# output details
# print(output_details)

def draw_rect2(image, box):
    h, w = image.shape[:2]
    y_min = int(max(1, (box[0] * h)))
    x_min = int(max(1, (box[1] * w)))
    y_max = int(min(h, (box[2] * h)))
    x_max = int(min(w, (box[3] * w)))
    
    # draw a rectangle on the image
    # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)


@detection_bp.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    

    file = request.files['image']
    image_bytes = file.read() # Read the uploaded image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)    

    # # Read the image file and preprocess it
    # img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    new_img = cv2.resize(img, (448, 448))


    interpreter.set_tensor(input_details[0]['index'], [new_img])

    interpreter.invoke()

    rect = interpreter.get_tensor(
        output_details[0]['index'])
    score = interpreter.get_tensor(
        output_details[2]['index'])

    print("ScoreA: ", score)
    print("Score: ", score[0])
    # print("Rect: ", rect[0])
    # print("img: ", img)

    if score[0] > 20.0:
        print("Image Pass: ")
        draw_rect2(new_img, rect[0])
        # buffer = cv2.imencode('.jpg', new_img)
        # print("Encoded img: ", buffer)
        # image_base64 = base64.b64encode(buffer).decode('utf-8')
        # Assuming `new_img` is your image data

    retval, buffer = cv2.imencode('.jpg', new_img)
    if retval:
        # Encoding successful, convert the buffer to base64
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        # print("Encoded img: ", image_base64)
    else:
        print("Failed to encode image")
        # Alternatively, you can use binascii.b2a_base64() for encoding
        # But ensure to pass only the image data (buffer), not the tuple
        encoded_image = binascii.b2a_base64(buffer).decode('utf-8')
        print("Encoded img: ", encoded_image)  
        


    # for index, score in enumerate(scores[0]):
    #     if score > 0.5:
    #       draw_rect(new_img, rects[0][index])
    #       encoded_img = cv2.imencode('.jpg', img)
    #       image_base64 = base64.b64encode(encoded_img).decode('utf-8')

    return render_template('index.html', result_image=encoded_image)
    # return jsonify({'detections': "score"})
