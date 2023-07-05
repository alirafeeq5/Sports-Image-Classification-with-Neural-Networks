from pathlib import Path

from tensorflow.keras import callbacks, Sequential
from tensorflow.keras.layers import (
    Rescaling, GaussianNoise, RandomFlip, RandomRotation,
    RandomContrast, RandomTranslation
)

from models import SqueezedXception
from utils import load_images, write_predictions


if __name__ == '__main__':
    DATA_PATH = Path('Data')
    TRAIN_DATA_PATH = DATA_PATH / 'Train'
    TEST_DATA_PATH = DATA_PATH / 'Test'

    _, dataset = load_images(TRAIN_DATA_PATH)
    dataset = dataset.shuffle(buffer_size=1700)

    validation_size = 400
    validation_dataset = dataset.take(validation_size)
    train_dataset = dataset.skip(validation_size)

    train_dataset = train_dataset.batch(32)
    validation_dataset = validation_dataset.batch(1)

    model = Sequential([
        Rescaling(1. / 255),
        GaussianNoise(0.4),
        RandomFlip("horizontal"),
        RandomTranslation(0.4, 0.6),
        RandomRotation(0.6),
        RandomContrast(0.4),
    ])

    model = SqueezedXception(model=model)

    callbacks = [
        callbacks.ReduceLROnPlateau(
            patience=7,
            factor=0.5
        ),
    ]

    model.fit(train_dataset, epochs=60, validation_data=validation_dataset, callbacks=callbacks)

    test_file_names, test_dataset = load_images(TEST_DATA_PATH, include_labels=False)
    predictions = model.predict(test_dataset.batch(1))

    write_predictions('predictions.csv', test_file_names, predictions)

    model.save('model.h5')
