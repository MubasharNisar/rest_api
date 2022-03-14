from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate
import cv2
import torchvision.transforms as T
from PIL import Image as pil_img
import numpy as np
import flask
from flask import render_template
from flask import request
import io
import pdb


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"].read()
        image = pil_img.open(io.BytesIO(image_file))
        image = T.ToTensor()(image)
        image = Image(image)
        if image:
            # CLI = cli.Prediction(image_file,paper_size).pred_result()
            result = load_model(image)
            pred = 'Predction: '+str(result)
            return render_template("index.html", pred='{}'.format(pred))
    return render_template("index.html", prediction=0)


def load_model(image):
    global model
    model = load_learner('', 'Model_densenet121_new.pkl')
    return model.predict(image)[0]


if __name__ == "__main__":
    app.run(port=5000, debug=True)
