import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, AveragePooling2D, ZeroPadding2D, Flatten, Dropout

#The input is a 32 32 pixel image.
input_img = (32,32,1)

num_of_classes = 10

def LENet5():
    inputs = keras.Input(shape=(32, 32, 1))

    # Layer C1 is a convolutional layer with six feature maps.
    # Each unit in each feature map is connected to a 5x5 neighborhood in the input.
    C1 = Conv2D(filters=6, kernel_size=(5, 5), strides=1)(inputs)
    # The size of the feature maps is 28x28.

    # which performs a local averaging and a subsampling.
    # Layer S2 is a subsampling layer with six feature maps of size 14x14.
    # Each unit in each feature map is connected to a 2x2 neighborhood in the corresponding feature map in C1.
    # The 2x2 receptive fields are nonoverlapping, therefore feature maps in S2 have half the number of rows and column as feature maps in C1.
    S2 = AveragePooling2D(pool_size=(2, 2), strides=2)(C1)

    # Layer C3 is a convolutional layer with 16 feature maps.
    # Each unit in each feature map is connected to several 5x5 neighborhoods at identical locations in a subset of S2’s feature maps.
    C3 = Conv2D(filters=16, kernel_size=(5, 5))(S2)

    # Layer S4 is a subsampling layer with 16 feature maps of size 5x5.
    S4 = AveragePooling2D(pool_size=(2, 2), strides=2)(C3)

    # Layer C5 is a convolutional layer with 120 feature maps.
    # Each unit is connected to a 5x5 neighborhood on all 16 of S4’s feature maps.
    C5 = Conv2D(filters=120, kernel_size=(5, 5))(S4)
    # Here, because the size of S4 is also 5x5, the size of C5’s feature maps is 1x1.
    # C5 is labeled as a convolutional layer, instead of a fully connected layer,
    # because if input were made bigger, the feature map dimension would be larger than 1x1.

    # Layer F6 contains 84 units.
    F6 = Dense(84)(C5)

    # The output layer is composed of Euclidean RBF units, one for each class, with 84 inputs each.
    F7 = Dense(10)(F6)

    outputs = Activation('softmax')(F7)

    return keras.Model(inputs=inputs, outputs=outputs)

model = LENet5()
model.summary()















