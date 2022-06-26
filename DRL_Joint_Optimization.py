import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os

Epochs = 50
Count = 5
Batch_size = 1024
TxA_loss = []
TxB_loss = []
RxA_loss = []
RxB_loss = []
noise_variance = 1e-3


class Channel:
    def __init__(self, noise_variance = noise_variance):
        self._x = np.array([])
        self._y = np.array([])
        self.noise_variance = noise_variance

    @property    #x.getter
    def x(self):
        return self._x

    @x.setter
    def x(self, val):
        self._x = val
        self._y = self._x + np.random.normal(loc=0,
                                             scale=np.sqrt(self.noise_variance)/2,
                                             size=self._x.shape)

    @property     #y.getter
    def y(self):
        return self._y


class Transmitter:
    def __init__(self, Nf=4, perturbation_variance=1e-4):
        self.Nf = Nf
        self.perturbation_variance = perturbation_variance

        self._x = np.array([])
        self._W = np.ones(self._x.shape + (2 * Nf, ))

        # Transmitter keras model
        tx_input = keras.layers.Input((1,), name='tx_input')
        x = keras.layers.BatchNormalization()(tx_input)
        x = keras.layers.Dense(units=5 * 2 * Nf, activation='elu', name='tx_10')(x)
        x = keras.layers.Dense(units=1 * 2 * Nf, activation='elu', name='tx_out')(x)
        self._model = keras.Model(inputs=tx_input, outputs=x)
        self._model.compile(loss='mse', optimizer='sgd')                                   #-------------------TO MODIFY----------------

    def train(self, channel: Channel, other_device):
        pass

    def map(self, m):
        self._x = self._model.predict(m)
        self._W = np.random.normal(loc=0,
                                   scale=np.sqrt(self.perturbation_variance) / 2,
                                   size=self._x.shape)
        return self._x

    @property
    def x(self):
        return self._x

    @property
    def x_p(self):
        return np.sqrt(1-self.perturbation_variance)*self._x + self._W

    @property
    def w(self):
        return self._W      #---------------------W.GETTER----------------------


class Receiver:
    def __init__(self, Nf=4):
        rx_input = keras.layers.Input((2 * Nf,), name='rx_input')
        # channel layer
        y = keras.layers.Dense(2 * Nf, activation='relu', name='rx_2')(rx_input)
        y = keras.layers.Dense(10 * Nf, activation='relu', name='rx_10')(y)
        pred = keras.layers.Dense(1, activation='relu', name='rx_output')(y)
        self._rx_model = keras.models.Model(inputs=rx_input, outputs=pred)
        self._rx_model.compile(loss='mse', optimizer='sgd')

    def train(self, other_device_transmitter: Transmitter, channel: Channel): 
        m = np.random.rand(Batch_size)
        x = other_device_transmitter.map(m)
        x_p = other_device_transmitter.x_p
        channel.x = x_p                                                            # transmitter.x_p    <----- this is different from the article, to be checked
        y = channel.y
        Hr = self._rx_model.fit(y, m, batch_size=Batch_size, epochs=1)
        return Hr

    def receive(self, symbols):
        return self._rx_model.predict(symbols)


class Device:
    def __init__(self):
        self.transmitter = Transmitter()
        self.receiver = Receiver()


    def train_tx(self, other_device, channel):
        r = np.random.rand(Batch_size)
        self.transmitter.map(r)

        # At this point we have x_p, just get it.
        channel.x = self.transmitter.x_p
        r_hat = other_device.receiver.receive(channel.y)

        channel.x = other_device.transmitter.map(r_hat)
        r_hat_hat = self.receiver.receive(channel.y-self.transmitter.w)
        Ht = self.transmitter._model.fit(r_hat_hat, r)
        return Ht

    def train_rx(self, other_device, channel):
        Hr = self.receiver.train(other_device.transmitter, channel)
        return Hr


if __name__ == '__main__':
    devA = Device()
    ch = Channel()
    devB = Device()

    
    for e in range(Epochs):
        for k in range(Count):
            print('dev A train tx...')
            HAt = devA.train_tx(devB, ch)
            TxA_loss.append(HAt.history['loss'])
            print('devB train rx...')
            HBr = devB.train_rx(devA, ch)
            RxB_loss.append(HBr.history['loss'])
        for k in range(Count):
            print('dev B train tx...')
            HBt = devB.train_tx(devA, ch)
            TxB_loss.append(HBt.history['loss'])
            print('devA train rx...')
            HAr = devA.train_rx(devB, ch)
            RxA_loss.append(HAr.history['loss'])
            
# Plots for training losses

import matplotlib.patches as mpatches

# Files

dirPath = os.getcwd()
TxA = '/Loss' + '_' + 'TxA' + '_' + str(Epochs) + '_' + str(Batch_size) + '_' + str(noise_variance) + '_' + '.txt'
Init = open(dirPath + '/results' + TxA, 'w')
Init.close()
with open(dirPath + '/results' + TxA, 'a') as DSTFile:
  DSTFile.write(str(np.squeeze(TxA_loss)))

dirPath = os.getcwd()
TxB = '/Loss' + '_' + 'TxB' + '_' + str(Epochs) + '_' + str(Batch_size) + '_' + str(noise_variance) + '_' + '.txt'
Init = open(dirPath + '/results' + TxB, 'w')
Init.close()
with open(dirPath + '/results' + TxB, 'a') as DSTFile:
  DSTFile.write(str(np.squeeze(TxB_loss)))

dirPath = os.getcwd()
RxA = '/Loss' + '_' + 'RxA' + '_' + str(Epochs) + '_' + str(Batch_size) + '_' + str(noise_variance) + '_' + '.txt'
Init = open(dirPath + '/results' + RxA, 'w')
Init.close()
with open(dirPath + '/results' + RxA, 'a') as DSTFile:
  DSTFile.write(str(np.squeeze(RxA_loss)))

dirPath = os.getcwd()
RxB = '/Loss' + '_' + 'RxB' + '_' + str(Epochs) + '_' + str(Batch_size) + '_' + str(noise_variance) + '_' + '.txt'
Init = open(dirPath + '/results' + RxB, 'w')
Init.close()
with open(dirPath + '/results' + RxB, 'a') as DSTFile:
  DSTFile.write(str(np.squeeze(RxB_loss)))

print(dirPath)

# Common reference for x axis
xl = np.arange(0,Epochs*Count) 

# Figure 0: Device A Training Loss (Transmitter)
plt.figure(0)
plt.title('Device A Training Loss (Transmitter)')
plt.xlabel('Epochs')
plt.ylabel('Device A Tx Loss')
plt.grid(True)
plt.plot(xl,np.squeeze(TxA_loss), 'r-')

# Figure 1: Device B Training Loss (Transmitter)
plt.figure(1)
plt.title('Device B Training Loss (Transmitter)')
plt.xlabel('Epochs')
plt.ylabel('Device B Tx Loss')
plt.grid(True)
plt.plot(xl,np.squeeze(TxB_loss), 'b-')

# Figure 2: Device A Training Loss (Receiver)
plt.figure(2)
plt.title('Device A Training Loss (Receiver)')
plt.xlabel('Epochs')
plt.ylabel('Device A Rx Loss')
plt.grid(True)
plt.plot(xl,np.squeeze(RxA_loss), 'r-')

# Figure 3: Device B Training Loss (Receiver)
plt.figure(3)
plt.title('Device B Training Loss (Receiver)')
plt.xlabel('Epochs')
plt.ylabel('Device B Rx Loss')
plt.grid(True)
plt.plot(xl,np.squeeze(RxB_loss), 'b-')

# Now, let's make a prediction to evaluate the performances of our trained NN

m = np.random.rand(Batch_size)
x = np.squeeze(devA.transmitter.map(m))
ch.x = x # Siamo sicuri che non ci vada la perturbazione?
m_hat = devB.receiver.receive(ch.y)
m_hat_squeezed = np.squeeze(m_hat)
print(m_hat_squeezed)

# Common reference for x axis
x = np.arange(0, Batch_size)

# Figure 4: True Messages vs Predicted Messages Scatterplot
plt.figure(4)
plt.title('True Messages vs Predicted Messages Scatterplot')
plt.xlabel('Message No')
plt.ylabel('True Messages / Predicted Messages')
red_patch = mpatches.Patch(color='red', label='True Messages')
green_patch = mpatches.Patch(color='green', label='Predicted Messages')
plt.legend(handles=[red_patch, green_patch])
plt.plot(x, m, 'ro')
plt.plot(x, m_hat_squeezed, 'go')

# Figure 5: Absolute Error Scatterplot
plt.figure(5)
plt.title('Figure 5: Absolute Error Scatterplot')
plt.xlabel('Message No')
plt.ylabel('Absolute Error')
plt.plot(x,np.abs(m-m_hat_squeezed), 'bo')

plt.show()
