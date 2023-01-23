from flask import Flask
from flask import render_template
from flask_wtf import FlaskForm
from wtforms import StringField, FileField, SubmitField, IntegerField, validators
from wtforms.validators import InputRequired, Length
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
import os
load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('APP_SECRET')
model = load_model('model.h5')



class PatientForm(FlaskForm):
    class Meta:
        csrf = False
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
        print(patient_form.name.data)
        print(patient_form.age.data)
    return render_template('result.html')


@app.route('/')
def home():
    patient_form = PatientForm()

    return render_template('index.html', form = patient_form)
if __name__ == '__main__':
    app.run(debug=True)
