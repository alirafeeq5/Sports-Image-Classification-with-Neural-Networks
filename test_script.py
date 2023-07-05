import sys
from pathlib import Path

from tensorflow.keras import models

from utils import load_images, write_predictions
from layers import SqueezedXceptionEntryRU, SqueezedXceptionMiddleRU


if len(sys.argv) != 2:
    print("Please provide the path to the test folder!")
    quit()

path = Path(sys.argv[1])
file_names, dataset = load_images(path, include_labels=False)

model = models.load_model(
    'model.h5',
    custom_objects={
        "SqueezedXceptionEntryRU": SqueezedXceptionEntryRU,
        "SqueezedXceptionMiddleRU": SqueezedXceptionMiddleRU
    }
)

predictions = model.predict(dataset.batch(1))

write_predictions('predictions.csv', file_names, predictions)
