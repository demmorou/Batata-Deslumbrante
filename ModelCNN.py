from keras import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
from keras.optimizers import Adam

'''

to run this model requires the installation of TensorFlow together with Keras. 
The TensorFlow backand is used.

-------------------------------------------------------------------------------------------------------------------------

O modelo Sequencial do Keras (Usando TensorFlow Banckend), é usado pois ele define que será adicionada uma camada por vez.
A parte mais importante são as camadas Convoluvionais (Conv2D).
Aqui estão definadas de 16 até 128 filtros quee usam 9 pesos.
Cada um dos pesos para transformar um pixel em uma média ponderada dele mesmo e seus oito vizinhos.
Como os mesmos nove pesos são usados em toda a imagem, a rede selecionará recursos úteis em todos os lugares. 
Como são apenas nove pesos, podemos empilhar muitas camadas convolucionais umas sobre as outras sem ficar sem memória/tempo.

-------------------------------------------------------------------------------------------------------------------------

A normalização em lotes (BatchNormalization), serve como um arranjo técnico para tornar o treino mais rápido.

-------------------------------------------------------------------------------------------------------------------------

Dropout é um método de regularização, onde a camada em questão substitui aleatoriamente uma proporção de seus pesos para 
zero para cada amostra de treinamento. 
Forçando a rede a aprender recursos de maneira distribuída, não confiando muito em um peso específico e, portanto, melhorando 
a generalização, pois é muito importante que a CNN seja capaz de generalizar o problema.

-------------------------------------------------------------------------------------------------------------------------


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
