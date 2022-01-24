import tensorflow as tf
import cv2

classes=[0,1,2,3,4,5,6,7,8,9]

model=tf.keras.models.load_model('mnist_cnn.h5')
# rnn
def testing():
    img = cv2.imread('image1.png',0)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img,(28,28))
    # 단층 인공신경망
    img = img.reshape(-1,28,28,1)
    #
    # img = img.reshape(-1,28,28,1)
    img = img.astype('float32')
    img = img/255.0

    pred = model.predict(img)
    return pred

