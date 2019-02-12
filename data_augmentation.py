from keras.preprocessing.image import ImageDataGenerator

shift = 0.1


def create_datagen():

    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=20,
        width_shift_range=shift,
        height_shift_range=shift,
        horizontal_flip=True,
        zca_whitening=False)

    return datagen
