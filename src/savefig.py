# Module to save activation figures with weights

from keras import models
import matplotlib.pyplot as plt

def saveActivations(fileprefix, model, x):

	layer_outputs = [layer.output for layer in model.layers[:4]]
	activation_model = models.Model(input=model.input, outputs=layer_outputs)
	activations = activation_model.predict([x])
	
	generateFig(activations[0:2], model.get_weights()[0], 4, fileprefix + '_first_.png')
	generateFig(activations[2:], model.get_weights()[2], 6, fileprefix + '_second_.png')
	

def generateFig(activations, weights, n_columns, output):

	fig = plt.figure(figsize=(n_columns,3))

	for i in range(0,n_columns):
		fig.add_subplot(3,n_columns,i+1)
		plt.axis('off')
		plt.imshow(weights[:,:,0,i])

	for i in range(0,2):
		for j in range(0,n_columns):
			fig.add_subplot(3,n_columns,(i+1)*n_columns+j+1)
			plt.axis('off')
			plt.imshow(activations[i][0,:,:,j])

	fig.savefig(output)
