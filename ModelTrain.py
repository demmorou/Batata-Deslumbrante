from keras.preprocessing.image import ImageDataGenerator
from ModelCNN import model_cnn

'''
	comments
	
Parâmentros 'ImageDataGenerator'
--------------------------------

	zoom_range:  é para um zoom aleatoriamente dentro de fotos.
	height_shift_range: desloca aleatoriamente imagens verticalmente (fração da altura total).
	width_shift_range: desloca aleatoriamente imagens horizontalmente (fração da largura total).
	rotation_range: é um valor em graus (0-180), um intervalo dentro do qual para girar aleatoriamente imagens.

Parâmetros 'flow_from_directory'
--------------------------------

	'data_set/train': pasta que contém os dados de treino (as imagens) devidamente separados de acordo com seus rótulos.
	target_size: tupla de inteiros (altura, largura), por padrão: (256, 256). As dimensões para as quais todas as imagens encontradas serão 	redimensionadas.
	batch_size: tamanho dos lotes de dados.
	class_mode: Determina o tipo de matrizes de rótulos da saída: "categorical" serão rótulos codificados em 2D, "binary" serão rótulos binários 		de 1D, "esparsos" serão rótulos inteiros de 1D, "input" serão imagens idênticas às imagens de entrada (usado principalmente para trabalhar 		com autoencodificadores).
	
Parâmetros 'fit_generator'
-------------------------------
	
	steps_per_epochs: Número total de etapas (lotes de amostras) antes de declarar uma época concluída e iniciar a próxima época. Ao treinar com 		tensores de entrada, como tensores de dados TensorFlow, o padrão None igual ao número de amostras em seu conjunto de dados dividido pelo 		tamanho do lote ou 1, se isso não puder ser determinado.
	epochs: Número de épocas para treinar o modelo. Uma época é uma iteração sobre o todo x e os y dados fornecidos. O modelo não é treinado para 		um número de iterações dadas por epochs, mas meramente até que a época do índice epochs seja alcançada.
	verbose: 0, 1 ou 2. Modo de verbosidade. 0 = silencioso, 1 = barra de progresso, 2 = uma linha por época.
	validation_data: tupla contendo (x_val, y_val) no qual irá avaliar a perda e qualquer métrica de modelo no final de cada época. O modelo não 		será treinado nesses dados. Isso irá substituir validation_split, que pode ser definido como 0 e 1 e funciona de maneira semelhante.
	
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

    train_set = train_datagen.flow_from_directory('data_set/train',
                                                  target_size=(150, 150),
                                                  batch_size=32,
                                                  class_mode='categorical')

    test_set = test_datagen.flow_from_directory('data_set/validation',
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
