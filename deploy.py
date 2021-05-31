import json

import numpy as np
from PIL import Image
import tensorflow as tf
from werkzeug.wrappers import Request, Response


model = None


def load_image(files):
    file_key = list(files.keys())[0]
    file = files.get(file_key)
    image = Image.open(file)
    image.load()
    image = image.resize((28, 28)).convert('L')
    image_data = np.array(image).reshape(1, 28, 28)
    image_data = image_data / 255.0
    return image_data


def predict(environ, start_response):
    # Load input image data from the HTTP request
    request = Request(environ)
    if not request.files:
        return Response('no file uploaded', 400)(environ, start_response)
    image = load_image(request.files)

    # Run the prediction
    global model
    if not model:
        model = tf.keras.models.load_model('model.h5')
    prediction = model.predict(image.reshape(1, 28, 28))

    # Return a JSON response
    result = {
        'digit': str(np.argmax(prediction))
    }

    print(json.dumps({'vh_metadata': result}))

    response = Response(json.dumps(result), content_type='application/json')
    return response(environ, start_response)


# If you call `python deploy.py`
# run a local server for testing
if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('0.0.0.0', 8000, predict)
