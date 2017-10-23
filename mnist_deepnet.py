## Code for SIBGRAPI'17 Tutorial 
## Everything you wanted to know about Deep Learning for Computer Vision but were afraid to ask 
## Moacir Ponti - Niteroi, RJ, Brazil - October 2017
## w/ Leonardo S. F. Ribeiro, Tiago S. Nazare, Tu Bui and John Collomosse

## MNIST Dataset
## Deep Dense Network with 2 hidden (intermediate) layers
## 1st Hidden Layer = 128 neurons
## 2nd Hidden Layer = 64 neurons
## It is a regular MLP network, but with ReLU activation functions

import tensorflow as tf

tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

from tensorflow.examples.tutorials.mnist import input_data as mnist_data

# definir matrizes e vetores
X = tf.placeholder(tf.float32, [None, 28, 28, 1]) # tamanho 28x28x1

# arquitetura com multiplas camadas
W1 = tf.Variable(tf.truncated_normal([784, 128], stddev=0.1))
b1 = tf.Variable(tf.ones([128])/10)

W2 = tf.Variable(tf.truncated_normal([128, 64], stddev=0.1))
b2 = tf.Variable(tf.ones([64])/10) 

W3 = tf.Variable(tf.truncated_normal([64,10], stddev=0.1))
b3 = tf.Variable(tf.ones([10])/10) # bias x 10 classes

Y = tf.placeholder(tf.float32, [None, 10]) # labels (ground-truth)

inicia = tf.global_variables_initializer() # instancia inicializacao

# vetor de entrada
X1 = tf.reshape(X, [-1, 784])

# representacoes intermediarias
X2 = tf.nn.relu(tf.matmul(X1, W1) + b1)
X3 = tf.nn.relu(tf.matmul(X2, W2) + b2)

# modelo que ira gerar as predicoes (camada de saida - softmax)
X4 = tf.matmul(X3, W3) + b3
Y_ = tf.nn.softmax(X4)

batchsize = 64

# loss-function entropia cruzada
# - Y * log(Y_)
entropia_cruzada = tf.nn.softmax_cross_entropy_with_logits(logits=X4, labels=Y)
entropia_cruzada = tf.reduce_mean(entropia_cruzada)*batchsize

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
	batX, batY = mnist.train.next_batch(batchsize)

	# organiza dicionario com pares (exemplo,rotulo)
	dados_treinamento = {X: batX, Y: batY}	

	# executa 1 iteracao com o minibatch	
	sess.run(treinamento, feed_dict=dados_treinamento)
	
	#ce = sess.run(entropia_cruzada, feed_dict=dados_treinamento)
	#rint(str(i) + " : Loss = " + str(ce))
	# computa acuracia de treinamento e a funcao de custo (loss)
	if (i%10 == 0):
		ac, ce = sess.run([acuracia, entropia_cruzada], feed_dict=dados_treinamento)
		print(str(i) + " Loss=" +str(ce) + " Training Acc=" + str(ac))


# verificar acuracia num conjunto de teste
dados_teste = {X: mnist.test.images, Y: mnist.test.labels}
ac = sess.run(acuracia, feed_dict=dados_teste)
print("Test Accuracy: " + str(ac))

