from keras.preprocessing.image import ImageDataGenerator
from ModelCNN import model_cnn

'''
    comments
'''


def model_train():

    model = model_cnn()

    train_datagen = ImageDataGenerator(zoom_range=0.1,
                                 height_shift_range=0.1,
                                 width_shift_range=0.1,
                                 rotation_range=10)

    test_datagen = ImageDataGenerator(zoom_range=0.1,
                                 height_shift_range=0.1,
                                 width_shift_range=0.1,
                                 rotation_range=10)

    train_set = train_datagen.flow_from_directory('/home/deusimar/Pictures/crop/crop-train',
                                                  target_size=(150, 150),
                                                  batch_size=32,
                                                  class_mode='categorical')

    test_set = test_datagen.flow_from_directory('/home/deusimar/Pictures/crop/crop-test',
                                                target_size=(150, 150),
                                                batch_size=32,
                                                class_mode='categorical')

    model.fit_generator(train_set,
                               steps_per_epoch=500,
                               epochs=100,  # Increase this when not on Kaggle kernel
                               verbose=2,  # 1 for ETA, 0 for silent
                               validation_data=test_set,  # For speed
                        )

    model.save_weights('weights002.h5')


if __name__ == '__main__':
    model_train()