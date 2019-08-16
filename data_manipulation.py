import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

train_images = []
test_images = []

with open('./dataset/food-101/meta/train.txt') as file:
    for image in file:
        image_class = image.split('/')[0]
        image = image[:-1] + '.jpg'
        train_images.append((image, image_class))

print(train_images)

with open('./dataset/food-101/meta/test.txt') as file:
    for image in file:
        image_class = image.split('/')[0]
        image = image[:-1] + '.jpg'
        test_images.append((image, image_class))

train_df = pd.DataFrame(train_images)
train_df.columns = ['filename', 'class']

print(train_df.head())

test_df = pd.DataFrame(test_images)
test_df.columns = ['filename', 'class']

train_datagen = ImageDataGenerator(rescale=1./255.,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1./255.)

train_gen = train_datagen.flow_from_dataframe(train_df, directory='./dataset/food-101/images/', subset='training',
                                              target_size=(200, 200))
valid_gen = train_datagen.flow_from_dataframe(train_df, directory='./dataset/food-101/images/', subset='validation',
                                              target_size=(200, 200))
test_gen = test_datagen.flow_from_dataframe(test_df, directory='./dataset/food-101/images/', target_size=(200, 200),
                                            shuffle=False)
