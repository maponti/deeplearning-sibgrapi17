## Code for SIBGRAPI'17 Tutorial 
## Everything you wanted to know about Deep Learning for Computer Vision but were afraid to ask 
## Moacir Ponti - Niteroi, RJ, Brazil - October 2017
## w/ Leonardo S. F. Ribeiro, Tiago S. Nazare, Tu Bui and John Collomosse

## MNIST Dataset
## Deep Convolutional Network with 2 hidden (intermediate) layers
## 1st Hidden Conv.Layer = 16 filters with size 5x5x1
## 2nd Hidden FC Layer = 128 neurons

import math
import tensorflow as tf
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

X = tf.placeholder(tf.float32, [None, 28, 28, 1]) # batch de imagens X

# cria e inicializa aleatoriamente os pesos com distribuicao normal e sigma=0.1

# primeira camada, 16 neuronios com filtros 5x5x1
W1 = tf.Variable(tf.truncated_normal([5,5,1,16], stddev=0.1))

# bias sao inicializados com valores fixos 1/10
B1 = tf.Variable(tf.ones([16])/10)

# segunda camada (densa), 64 neuronios
# como o filtro tinha 5x5 e iremos fazer um maxpool com stride 2, a imagem tera (12x12) x 16 feature maps
W2 = tf.Variable(tf.truncated_normal([12*12*16, 128], stddev=0.1))
B2 = tf.Variable(tf.ones([128])/10)

# terceira camada (densa) de saida com 10 neuronios
W3 = tf.Variable(tf.truncated_normal([128, 10], stddev=0.1))
B3 = tf.Variable(tf.ones([10])/10)

Y = tf.placeholder(tf.float32, [None, 10]) # classes das imagens em X

# learning rate
lr = tf.placeholder(tf.float32)
probkeep = tf.placeholder(tf.float32)

# entrada redimensionada
X1 = tf.reshape(X, [-1, 28, 28, 1])

# modelos das representacoes intermediarias
X2 = tf.nn.conv2d(X1,W1,strides=[1,1,1,1],padding="VALID")
X2 = tf.nn.relu(tf.nn.bias_add(X2, B1))
X2p = tf.layers.max_pooling2d(X2,2,2)
X2f = tf.contrib.layers.flatten(X2p)
#ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")

X3 = tf.matmul(X2f, W2)
X3 = tf.nn.relu(tf.nn.bias_add(X3, B2))
X3d = tf.nn.dropout(X3, probkeep)

# saida da rede (a.k.a. logits)
X4 = tf.nn.bias_add(tf.matmul(X3d, W3), B3)
# classificacao softmax
Y_ = tf.nn.softmax(X4)

# utilizamos uma funcao pronta no TF para calculo da entropia cruzada
entropia_cruzada = tf.nn.softmax_cross_entropy_with_logits(logits=X4, labels=Y)
entropia_cruzada = tf.reduce_mean(entropia_cruzada)*64

eh_correto = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
acuracia = tf.reduce_mean(tf.cast(eh_correto, tf.float32))

# otimizacao com taxa de aprendizado 0.0025
otimiza = tf.train.AdamOptimizer(lr)
treinamento = otimiza.minimize(entropia_cruzada)

inicia = tf.global_variables_initializer() # instancia inicializacao
sess = tf.Session() # instancia sessao
sess.run(inicia)    # executa sessao e inicializa
# baixa base de dados mnist
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# executa 2000 iteracoes
for i in range(2000):
    # carrega batch de 64 imagens (X) e suas classes (Y)
    batch_X, batch_Y = mnist.train.next_batch(64)

    # reduz a taxa de aprendizado
    max_lear_rate = 0.003
    min_lear_rate = 0.0001
    decay_speed = 2000.0 # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
    learn_rate = min_lear_rate + (max_lear_rate - min_lear_rate) * math.exp(-i/decay_speed)

    dados_trein={X: batch_X, Y: batch_Y, lr:learn_rate, probkeep:0.75}

    # treina com o batch atual
    sess.run(treinamento, feed_dict=dados_trein)

    # a cada 10 iteracoes, computa entropia-cruzada para acompanhar convergencia
    if (i%10 == 0):
       ac, ce = sess.run([acuracia, entropia_cruzada], feed_dict=dados_trein)
       print(str(i) + " Loss=" +str(ce) + " Training Acc=" + str(ac))


dados_teste={X: mnist.test.images, Y: mnist.test.labels, probkeep:1.0}
a,c = sess.run([acuracia, entropia_cruzada], feed_dict=dados_teste)
print("Test Accuracy: ",str(a))
print("- Loss: ",str(c))
