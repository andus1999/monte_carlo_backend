import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from ..resources import filepaths
import os


def get_db():
    try:
        app = firebase_admin.get_app('decryptor')

    except ValueError:
        cred = credentials.Certificate(os.path.join(os.path.dirname(__file__), filepaths.firebase_db_key))
        app = firebase_admin.initialize_app(cred, {
            'projectId': 'decryptor-329419'
        }, 'decryptor')
    db = firestore.client(app)
    return db


def upload_coin_data(coin_data):
    db = get_db()

    coin = coin_data['id']
    db.collection(u'coins').document(coin).set(coin_data, merge=True)


def upload_prediction_data(coin, model, data):
    db = get_db()

    db.collection(u'coins').document(coin).collection(u'predictions').document(model).set(data)


def upload_model_data(model_data):
    db = get_db()

    model = model_data['model_name']
    db.collection(u'models').document(model).set(model_data, merge=True)
