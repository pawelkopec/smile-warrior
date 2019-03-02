from keras.preprocessing.image import ImageDataGenerator


def create_datagen():

    shift = 0.1

    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=20,
        width_shift_range=shift,
        height_shift_range=shift,
        horizontal_flip=True,
        zca_whitening=False)

    return datagen
