import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from ..resources import filepaths
import os
from .. import utils


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


def upload_historical_data(coin, data):
    db = get_db()
    short_data = utils.data.get_historical_data(data)

    db.collection(u'coins').document(coin).collection(u'historical_data').document(u'historical_data').set(short_data)
