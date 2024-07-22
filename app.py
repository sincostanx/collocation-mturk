from flask import Flask, render_template, jsonify, send_from_directory, send_file
import os
from utils import get_mask, overlay_mask
import io
from PIL import Image

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
print(BASE_DIR)

# List of image paths
image_paths = [
    'dataset/rev1/abandon_factory_0001_Background.png',
    'dataset/rev1/cat.png',
    'data/images/a-cat/000000.png'
]

# Index to keep track of the current image
current_image_index = 0

@app.route('/')
def index():
    return render_template('index.html', image_path=image_paths[current_image_index])

@app.route('/ok', methods=['POST'])
def ok():
    print("ok")
    return switch_image()

@app.route('/wrong', methods=['POST'])
def wrong():
    print("wrong")
    return switch_image()

def switch_image():
    global current_image_index
    current_image_index = (current_image_index + 1) % len(image_paths)
    return jsonify({'next_image': image_paths[current_image_index]})

@app.route('/image/<path:filename>')
def get_image(filename):
    # directory, file = os.path.split(filename)
    # return send_from_directory(os.path.join(BASE_DIR, "..", directory), file)

    image_path = os.path.join(BASE_DIR, "..", filename)
    image = Image.open(image_path)

    mask_png_data = "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"
    mask = get_mask(mask_png_data)

    result = overlay_mask(image, mask, return_type="byteio")
    return send_file(io.BytesIO(result), mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
