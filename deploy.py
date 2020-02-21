import json

from matplotlib.image import imread
import numpy as np
import tensorflow as tf
from werkzeug.wrappers import Request, Response


class MnistPredict(object):
    def __init__(self):
        # Load the model
        self.model = tf.keras.models.load_model('model.h5')

    def load_image(self, files):
        file_key = list(files.keys())[0]
        file = files.get(file_key)
        return imread(file).reshape(1, 28, 28)

    def wsgi_app(self, environ, start_response):
        # Load input image data from the HTTP request
        request = Request(environ)
        if not request.files:
            return Response('no file uploaded', 400)(environ, start_response)
        image = self.load_image(request.files)

        # Run the prediction
        prediction = self.model.predict(image.reshape(1, 28, 28))

        # Return a JSON response
        result = {
            'digit': str(np.argmax(prediction))
        }
        response = Response(json.dumps(result), content_type='application/json')
        return response(environ, start_response)

    def __call__(self, environ, start_response):
        return self.wsgi_app(environ, start_response)


def create_app():
    return MnistPredict()


# If you call `python deploy.py`
# run a local server for testing
if __name__ == '__main__':
    from werkzeug.serving import run_simple
    app = create_app()
    run_simple('0.0.0.0', 8000, app)
