from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pylab as plt
import numpy as np

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

for layer in base_model.layers:
  layer.trainable=False

#Create a global average pooling layer
x = GlobalAveragePooling2D()(base_model.output)

#Add a fully connected layer
x = Dense(1024, activation='relu')(x)

#Add a classification layer with 2 labels
predictions = Dense(2, activation='softmax')(x)

#This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

model.summary()
model.compile(optimizer=Adam(learning_rate=0.01),
               loss='categorical_crossentropy',
                 metrics=['accuracy'])

image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(
        '/home/youmna/Documents/Detecto Project/fruits_data_advanced/data/train/',
        target_size=(image_size, image_size),
        batch_size=1,
        class_mode='categorical')

test_generator = data_generator.flow_from_directory(
        '/home/youmna/Documents/Detecto Project/fruits_data_advanced/data/test/',
        target_size=(image_size, image_size),
        batch_size=1,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        '/home/youmna/Documents/Detecto Project/fruits_data_advanced/data/validation/',
        target_size=(image_size, image_size),
        batch_size=1,
        class_mode='categorical')

history = model.fit_generator(train_generator,
                    validation_data = validation_generator,
                    steps_per_epoch = train_generator.n//train_generator.batch_size,
                    validation_steps = validation_generator.n//validation_generator.batch_size,
                    epochs=5, verbose = True)

#Extract the loss values from the history
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

params = {
   'axes.labelsize': 18,
   'font.size': 18,
   'legend.fontsize': 18,
   'xtick.labelsize': 14,
   'ytick.labelsize': 14,
   'text.usetex': False,
   'figure.figsize': [9, 6]
   }
plt.rcParams.update(params)


#Create a plot to visualize the training and validation loss
plt.plot(range(1, len(train_acc) + 1), train_acc, label='Training Accuracy', marker='.')
plt.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy', marker='.')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(False)
plt.show()

#Load and preprocess the image
img_path = '/home/youmna/Documents/Detecto Project/banana.webp'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

#Make predictions
predictions = model.predict(x)
plt.imshow(img)
print(predictions)

#Calculate the success rates
total_apples = 0
correct_apples = 0
total_bananas = 0
correct_bananas = 0

apple_label = 0
banana_label = 1

step_counter = 0

for x_batch, y_batch in test_generator:
    predictions = model.predict(x_batch)

    for i, prediction in enumerate(predictions):
        actual_label = np.argmax(y_batch[i])
        predicted_label = np.argmax(prediction)

        if actual_label == apple_label:
            total_apples += 1
            if predicted_label == apple_label:
                correct_apples += 1

        elif actual_label == banana_label:
            total_bananas += 1
            if predicted_label == banana_label:
                correct_bananas += 1

    step_counter += 1

    if step_counter == 10:
        break

N1 = correct_apples
N2 = correct_bananas
success_rate = (N1 + N2) / (total_apples + total_bananas)

print(f"Success Rate: {success_rate:.2f}")
print(f"Correct Apples: {correct_apples}/{total_apples}, Correct Bananas: {correct_bananas}/{total_bananas}")

#Evaluate the model
test_loss_accuracy = model.evaluate(test_generator)
print(f"Test Loss and Accuracy: {test_loss_accuracy}")

#Save the model
model.save('model_resnet50.keras')