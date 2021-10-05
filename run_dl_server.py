from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate
import cv2
import torchvision.transforms as T
from PIL import Image as pil_img
import numpy as np
import flask
import io
import pdb
# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None


def load_model():
	global model
	model = load_learner('' , 'Model_densenet121_new.pkl')

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}
	

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = pil_img.open(io.BytesIO(image))
			image = T.ToTensor()(image)
			image = Image(image)
			data["prediction"] = []
			data["prediction"].append(str(model.predict(image)[0]))
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	app.run()