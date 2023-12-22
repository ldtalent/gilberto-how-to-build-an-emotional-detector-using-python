from keras.layers import Activation, Convolution2D, Dropout, GlobalAveragePooling2D, AveragePooling2D, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Input

def simple_CNN(input_shape, num_classes, filter_sizes=(16, 32, 64, 128, 256), kernel_sizes=((7, 7), (5, 5), (3, 3))):
    model = Sequential()
    
    for i, (filters, kernel_size) in enumerate(zip(filter_sizes, kernel_sizes), 1):
        model.add(Convolution2D(filters=filters, kernel_size=kernel_size, padding='same', input_shape=input_shape, name=f'conv_{i}'))
        model.add(BatchNormalization(name=f'batch_norm_{i}'))
        model.add(Activation('relu', name=f'activation_{i}'))
        model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.5, name=f'dropout_{i}'))

    model.add(Convolution2D(filters=num_classes, kernel_size=(3, 3), padding='same'))
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax', name='predictions'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    input_shape = (48, 48, 1)
    num_classes = 7
    model = simple_CNN(input_shape, num_classes)
    model.summary()
