## Code for SIBGRAPI'17 Tutorial 
## Everything you wanted to know about Deep Learning for Computer Vision but were afraid to ask 
## Moacir Ponti - Niteroi, RJ, Brazil - October 2017
## w/ Leonardo S. F. Ribeiro, Tiago S. Nazare, Tu Bui and John Collomosse

## MNIST Dataset
## Shallow Network, with input -> output layers

import tensorflow as tf

tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

from tensorflow.examples.tutorials.mnist import input_data as mnist_data

# definir matrizes e vetores
X = tf.placeholder(tf.float32, [None, 28, 28, 1]) # tamanho 28x28x1

W = tf.Variable(tf.zeros([784,10])) # pesos 784 features x 10 classes
b = tf.Variable(tf.zeros([10])) # bias x 10 classes

Y = tf.placeholder(tf.float32, [None, 10]) # labels (ground-truth)

inicia = tf.global_variables_initializer() # instancia inicializacao

# modelo que ira gerar as predicoes
# Y = funcao(WX+b) = softmax(WX+b)

Y_ = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) + b)

# loss-function entropia cruzada
# - Y * log(Y_)
entropia_cruzada = -tf.reduce_sum(Y * tf.log(Y_ + 0.0001))

# acuracia
eh_correto = tf.equal(tf.argmax(Y_,1), tf.argmax(Y,1))
acuracia = tf.reduce_mean(tf.cast(eh_correto, tf.float32))

# define metodo otimizador (learning rate)
otimizador = tf.train.GradientDescentOptimizer(0.003)
treinamento = otimizador.minimize(entropia_cruzada)

sess = tf.Session() # instancia sessao (para executar em CPU/GPU)
sess.run(inicia)

# baixa base de dados mnist
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# iteracoes da rede (feed-forward + backpropagation)
for i in range(2000):
	#carrego minibatch de dados e respectivas classes
	batX, batY = mnist.train.next_batch(64)

	# organiza dicionario com pares (exemplo,rotulo)
	dados_treinamento = {X: batX, Y: batY}	

	# executa 1 iteracao com o minibatch	
	sess.run(treinamento, feed_dict=dados_treinamento)
	
	# computa acuracia de treinamento e a funcao de custo (loss)
	if (i%10 == 0):
		ac, ce = sess.run([acuracia, entropia_cruzada], feed_dict=dados_treinamento)
		print(str(i) + " Loss=" +str(ce) + " Training Acc=" + str(ac))


# verificar acuracia num conjunto de teste
dados_teste = {X: mnist.test.images, Y: mnist.test.labels}
ac = sess.run(acuracia, feed_dict=dados_teste)
print("Test Accuracy: " + str(ac))

