from data_manipulation import train_gen, valid_gen, test_gen
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import ceil
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import keras.regularizers as regularizers
import keras.optimizers as optimizers

inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

model = Sequential()

model.add(inception)
model.add(GlobalAveragePooling2D())
# model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(101, activation='softmax', kernel_regularizer=regularizers.l2(0.005)))

opt = optimizers.Adam(lr=0.0001)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
STEP_SIZE_VALID = valid_gen.n // valid_gen.batch_size
STEP_SIZE_TEST = ceil(test_gen.n / test_gen.batch_size)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15)
csv_logger = CSVLogger('food101_inceptionv3.log')
checkpoint = ModelCheckpoint('food101_inceptionv3_best.h5', save_best_only=True)

history = model.fit_generator(generator=train_gen,
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=valid_gen,
                              validation_steps=STEP_SIZE_VALID,
                              epochs=100000,
                              callbacks=[early_stopping, csv_logger, checkpoint]
                              )

model.evaluate_generator(generator=valid_gen, steps=STEP_SIZE_VALID)

test_gen.reset()
pred = model.predict_generator(test_gen,
                               steps=STEP_SIZE_TEST,
                               verbose=1)

print(pred)

predicted_class_indices = np.argmax(pred, axis=1)

labels = train_gen.class_indices
print(labels)
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames = test_gen.filenames
results = pd.DataFrame({"Filename": filenames,
                        "Predictions": predictions})

results.to_csv("results_inceptionv3.csv", index=False)

model.save('food101_inceptionv3.h5')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'test'])
plt.show()
