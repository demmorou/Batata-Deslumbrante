from keras.preprocessing import image
from ModelCNN import model_cnn
import numpy as np
import itertools
import matplotlib.pyplot as plt


def prediction():

    import glob as g

    model = model_cnn()
    model.load_weights('weights001.h5')
    pred = None

    y_true = []
    y_pred = []

    list_normal = g.glob('/home/deusimar/Pictures/crop/crop-test/normal/*.png')

    for index, value in enumerate(list_normal):
        test_image = image.load_img(value, target_size=(150, 150))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        pred = model.predict_classes(test_image, verbose=2)[0]
        y_pred.append(pred)
        y_true.append(1)

    l = g.glob('/home/deusimar/Pictures/crop/crop-test/avancado/*.png') # load images for predict (advanced)
    for indice, value in enumerate(l):
        test_image = image.load_img(value, target_size=(150, 150))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        pred = model.predict_classes(test_image, verbose=2)[0]
        y_pred.append(pred)
        y_true.append(0)

    # return the values of prediction and values true
    return y_true, y_pred


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting normalize=True.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def accuracy(vn, fp, fn, vp):
    return (vp + vn) / (vp + vn + fp + fn)


def confusio_matrix():

    y_true, y_pred = prediction()

    from sklearn.metrics import confusion_matrix

    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)

    vn, fp, fn, vp = confusion_matrix(y_true, y_pred).ravel()
    a = (accuracy(vn, fp, fn, vp) * 100)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['COM DR', 'SEM DR'],
                          title='Confusion Matrix - Accuracy: %.2f' % a)

    plt.show()

if __name__ == '__main__':
    confusio_matrix()