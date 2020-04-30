import numpy as np
from absl import app
from tensorflow_core.python.keras.models import load_model, save_model

from data_reader import DataInput
from layer import Attention_COSTUM
from model import get_model


def main(_):
    data = DataInput('data.json')
    X_train, Y_train, X_test, Y_test = data.get_train_test_data()

    model = get_model(data.word_list, data.label_len)
    #model.summary()
    attlstm_history = model.fit(X_train[:128],
                                Y_train[:128],
                                batch_size=128,
                                epochs=1,
                                validation_data=(X_test, Y_test))

    print(model.weights[0])
    #model.save('path_to_saved_model', save_format='tf')
    save_model(model, 'model.h5')
    del model
    model = load_model('model.h5', custom_objects={'attention_costum': Attention_COSTUM})
    print(model.weights[0])


if __name__ == '__main__':
    app.run(main)
