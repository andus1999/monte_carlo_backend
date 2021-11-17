import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from monte_carlo.resources import filepaths
from . import data
from ..settings import models
import os


def get_db():
    try:
        app = firebase_admin.get_app('decryptor')

    except ValueError:
        cred = credentials.Certificate(os.path.join(os.path.dirname(__file__), filepaths.firebase_db_key))
        app = firebase_admin.initialize_app(cred, {
            'storageBucket': 'decryptor-329419.appspot.com',
            'projectId': 'decryptor-329419'
        }, 'decryptor')
    db = firestore.client(app)
    return db


def upload_prediction_data(prediction_objects):
    db = get_db()
    prediction_data = data.get_prediction_data(prediction_objects)

    model = prediction_objects[-1]['model']
    db.collection(u'predictions').document(models.get_id(model)).set(prediction_data)
    if model == models.main_model:
        db.collection(u'predictions').document(u'main').set(prediction_data)


def upload_model_data(prediction_objects):
    db = get_db()

    model_data = data.get_model_data(prediction_objects)

    model = prediction_objects[-1]['model']
    db.collection(u'models').document(models.get_id(model)).set(model_data)
    if model == models.main_model:
        db.collection(u'models').document(u'main').set(model_data)
