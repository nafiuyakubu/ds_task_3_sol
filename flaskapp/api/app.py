from flask import Flask, request, render_template, jsonify, abort, make_response
from flask_cors import CORS
# Import all needed blueprints
# from src.detection.detection_routes import detection_bp

app = Flask(__name__, template_folder='templates')
CORS(app)  # This will allow CORS for all routes
# Register blueprints
# app.register_blueprint(detection_bp)

@app.route('/')
def index():
    return render_template('index.html', result_image=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5000"), debug=True) 