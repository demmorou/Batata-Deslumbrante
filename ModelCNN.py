from keras import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
from keras.optimizers import Adam

'''

to run this model requires the installation of TensorFlow together with Keras. 
The TensorFlow backand is used.

'''

def model_cnn():

	# Modelo Sequencial do Keras (backand TensorFlow)
    
	model = Sequential() 
	
	# Add Convolutional Layer 
    	model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(150, 150, 3)))
	
	# Add Batch Normalization
    	model.add(BatchNormalization())
	
	# Add Convolutional Layer
    	model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    	
	# Add Batch Normalization
	model.add(BatchNormalization())

	# Add Pooling Layer
    	model.add(MaxPool2D(strides=(2, 2)))

	# Add Dropout layer
    	model.add(Dropout(0.25))

	# Add Convolutional layer 
    	model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
	
	# Add Batch Normalization
    	model.add(BatchNormalization())
	
	# Add Convolutional layer
    	model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    
	# Add Batch Normalization
	model.add(BatchNormalization())

	# Add Pooling Layer    	
	model.add(MaxPool2D(strides=(2, 2)))

	# Add Dropout Layer    	
	model.add(Dropout(0.25))

	# Add Convolutional layer
    	model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    	
	# Add Batch Normalization
	model.add(BatchNormalization())

	# Add Convolutional layer
	model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    	
	# Add Batch Normalization
	model.add(BatchNormalization())

	# Add Pooling Layer
    	model.add(MaxPool2D(strides=(2, 2)))
    
	# Add Dropout Layer	
	model.add(Dropout(0.25))

	# Add Convolutional layer    	
	model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
	
	# Add Batch Normalization
    	model.add(BatchNormalization())

	# Add Convolutional layer  
    	model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    
	# Add Batch Normalization
	model.add(BatchNormalization())

    	# Add Pooling Layer
	model.add(MaxPool2D(strides=(2, 2)))
	
	# Add Dropout Layer
    	model.add(Dropout(0.25))

	# Connecting all layers previous
    	model.add(Flatten())
	
	# Add Dense Layer    
	model.add(Dense(512, activation='relu'))
	
	# Add Dropout Layer    	
	model.add(Dropout(0.25))

	# Add Dense Layer
	model.add(Dense(1024, activation='relu'))
    	
	# Add Dropout Layer 
	model.add(Dropout(0.5))
	
	# Last Dense Layer (2 posible predictions)
    	model.add(Dense(2, activation='softmax'))

	# Compiling the model with loss = 'categorical_crossebtropy' and optimizer = 'Adam' and metrics = 'Accuracy'     	
	model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=["accuracy"])

    return model
