from keras.preprocessing.image import ImageDataGenerator

shift = 0.1


def create_datagen():
    """
    Function creates ImageDataGenerator used in process of augmentation Smile-Warrior dataset to increase accuracy of
    trained neural network. Arguments of ImageDataGenerator can be adjusted below.

    Example:

            datagen = create_datagen()

    :return:
        ImageDataGenerator: Class generating batches of tensor image data with real-time data augmentation
    """

    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=20,
        width_shift_range=shift,
        height_shift_range=shift,
        horizontal_flip=True,
        zca_whitening=False)

    return datagen
