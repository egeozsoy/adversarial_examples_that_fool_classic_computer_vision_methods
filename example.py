from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from load_cifar10_data import get_data
from foolbox.models import KerasModel
from foolbox.criteria import TargetClass
import foolbox
import numpy as np
import utils

#create model
model = Sequential()
#add model layers
model.add(Conv2D(4, kernel_size=3, activation='relu', input_shape=(32,32,3)))
model.add(Conv2D(2, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
data = get_data()
X_train = data[b'data']
y_train = data[b'labels']
#model.fit(X_train, y_train, epochs=1)

image = np.float64(X_train[0:1])
label = np.argmax(model.predict(image))
label = np.float64(label)
adversarial = None
while adversarial is None:
    fmodel = KerasModel(model, bounds=(0, 255), )
    attack = foolbox.attacks.BoundaryAttack(fmodel, criterion=TargetClass(0))
    adversarial = attack(image[0], label,  iterations=500)

print(np.argmax(model.predict(image)))
print(np.argmax(model.predict(np.array([adversarial]))))

utils.plot_result(image, adversarial)