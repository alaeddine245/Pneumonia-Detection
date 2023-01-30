from flask import Flask
from flask import render_template
from flask_wtf import FlaskForm
from wtforms import StringField, FileField, SubmitField, IntegerField, validators
from wtforms.validators import InputRequired, Length
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import os
import base64
import numpy as np
import cv2
load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('APP_SECRET')
model = load_model('model.h5')



class PatientForm(FlaskForm):
    name = StringField(
        validators=[InputRequired(), Length(min=2, max=20)])
    age = IntegerField(validators=[InputRequired()])
    hospital = StringField(validators=[InputRequired(), Length(min=2, max=20)])
    image = FileField()
    submit = SubmitField()


@app.route('/predict', methods=['POST'])
def predict():
    patient_form = PatientForm()
    if patient_form.validate():
        im_bytes = patient_form.image.data.read()
        img = np.frombuffer(im_bytes, dtype=np.uint8)
        img_cv = cv2.imdecode(img, flags=cv2.IMREAD_COLOR)
        img_resized= cv2.resize(img_cv,dsize=(224,224), interpolation = cv2.INTER_CUBIC)
        np_image_data = np.asarray(img_resized)
        np_final = np.expand_dims(np_image_data,axis=0)
        print(np_final.shape)
        prediction=model.predict({'input_8': np_final})
        result = ""
        if (prediction > 0.5):
            result = "Pneumonia"
        else:
            result = "Not Pneumonia"
    return render_template('result.html', result = result, image=base64.b64encode(im_bytes).decode('ascii'))


@app.route('/')
def home():
    print(model.summary())
    patient_form = PatientForm()

    return render_template('index.html', form = patient_form)
if __name__ == '__main__':
    app.run(debug=True)
