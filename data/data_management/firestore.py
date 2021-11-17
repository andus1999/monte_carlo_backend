import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from ..resources import filepaths
from .. import data_management
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


def upload_historical_data(coin, data):
    db = get_db()

    short_data = data_management.data.get_coin_data_json(data)
    db.collection(u'coins').document(coin).set(short_data)
