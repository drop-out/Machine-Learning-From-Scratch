
import numpy as np
from abc import ABCMeta, abstractmethod


class loss(metaclass=ABCMeta):
    '''
    The abstract base class for loss function.
    For each loss, the gradient should be specified.
    '''
    def obj(self,pred,true):
        pass
    
    def gradient(self,pred,true):
        pass
        
class mse(loss):
    '''Loss function for mse.'''
    def obj(self,pred,true):
        return np.square(pred-true).mean()/2
    
    def gradient(self,pred,true):
        return pred-true
    
class log_loss(loss):
    '''Loss fucntion for log loss.'''
    def obj(self,pred,true):
        return (-np.multiply(true,np.log(pred))-np.multiply(1-y,np.log(1-pred))).mean()
    
    def gradient(self,pred,true):
        return -np.multiply(true,1/pred)+np.multiply(1-true,1/(1-pred))


class act(metaclass=ABCMeta):
    '''
    The abstract base class for activation function.
    For each loss, 
    the functions used for forward and backward propagation are specified respectively.
    The two functions take same inputs.
    The forward function would return the values after the transformation.
    The backward function would return the derivative musk at this layer.
    '''
    def forward(self,matrix):
        pass
    
    def backward(self,matrix):
        pass
        
class linear(act):
    '''Linear activation function.'''
    def forward(self,matrix):
        return matrix
    
    def backward(self,matrix):
        return np.ones_like(matrix)
    
class relu(act):
    '''Rectified linear units.'''
    def forward(self,matrix):
        return np.multiply(matrix>0,matrix)
    
    def backward(self,matrix):
        return 1*(matrix>0)
        
class logistic(act):
    '''Logistic transformation'''
    def forward(self,matrix):
        return 1/(1+np.exp(-matrix)+0.000001)
    
    def backward(self,matrix):
        return np.multiply(self.forward(matrix),1-self.forward(matrix))


class MLP(object):
    '''
    Parameters:
    ----------
    n_hidden_units: Number of units in the hidden layer.
    batch_size: Number of data points used in each gradient step.
    n_epochs: Number of epochs.
              Note that this determines the number of epochs (how many times each data point will be used),
              not the number of gradient steps.
    learning_rate: The learning rate of gradient descent.
    momentum: Momentum for gradient descent update. (Between 0 and 1.)
    weight_decay: Coeffecients for L2 regularization. (Also known as weight decay.)
    activation: Activation function for the hidden layer.
                'relu' for rectified linear units.
                'logistic' for sigmoid activation.
                'linear' for linear activation
    loss: Loss function.
          'mse' for regression task
          'log_loss' for classfication task.
    '''
        
    def __init__(self,
                 n_hidden_units=100,
                 batch_size=200,
                 n_epochs=200,
                 learning_rate=0.01,
                 momentum=0.9,
                 weight_decay=0.0001,
                 activation='relu',
                 loss='mse'):

        self.n_hidden_units=n_hidden_units
        self.batch_size=batch_size
        self.n_epochs=n_epochs
        self.learning_rate=learning_rate
        self.momentum=momentum
        self.weight_decay=weight_decay

        #activation (This is the activation function for the hidden layer.)
        if activation=='relu':
            self.act1=relu()
        elif activation=='logistic':
            self.act1=logistic()
        elif activation=='linear':
            self.act1=linear()
        else:
            self.act1=activation

        #loss (Note that the activation function for the output layer is determined by the loss.)
        if loss=='mse':
            self.loss=mse()
            self.act2=linear()
        elif loss=='log_loss':
            self.loss=log_loss()
            self.act2=logistic()
        else:
            self.loss=loss[0]
            self.act2=loss[1]

    def forward(self):
        self.layer1=self.W1*self.X+self.b1
        self.layer1act=self.act1.forward(self.layer1)
        self.score=self.W2*self.layer1act+self.b2
        self.pred=self.act2.forward(self.score)

    def backward(self):
        self.dpred=self.loss.gradient(self.pred,self.true)
        self.dscore=np.multiply(self.dpred,self.act2.backward(self.score))
        self.dlayer1act=self.W2.T*self.dscore
        self.dlayer1=np.multiply(self.dlayer1act,self.act1.backward(self.layer1))
        
        self.dW1=(self.dlayer1*self.X.T-self.weight_decay*self.W1)/self.batch_size
        self.db1=np.sum(self.dlayer1,axis=1)/self.batch_size
        self.dW2=(self.dscore*self.layer1act.T-self.weight_decay*self.W2)/self.batch_size
        self.db2=np.sum(self.dscore,axis=1)/self.batch_size

    def update_weights(self):
        #calculate moving average gradients(momentum)
        self.tW1 = self.momentum*self.tW1+(1-self.momentum)*self.dW1
        self.tb1 = self.momentum*self.tb1+(1-self.momentum)*self.db1
        self.tW2 = self.momentum*self.tW2+(1-self.momentum)*self.dW2
        self.tb2 = self.momentum*self.tb2+(1-self.momentum)*self.db2

        #update weights
        self.W1-=self.tW1*self.learning_rate
        self.b1-=self.tb1*self.learning_rate
        self.W2-=self.tW2*self.learning_rate
        self.b2-=self.tb2*self.learning_rate
        
    def fit(self,train,target):
        #turn the inputs into matrices.
        train=np.matrix(train).T
        target=np.matrix(target.reshape(-1,1)).T
        
        #parameters
        n_features=train.shape[0]
        n_obs=train.shape[1]

        #weights initialization
        s1=np.sqrt(6/(n_features+self.n_hidden_units))
        s2=np.sqrt(6/(1+self.n_hidden_units))
        self.W1=np.matrix(np.random.uniform(-s1,s1,[self.n_hidden_units,n_features]))
        self.b1=np.matrix(np.random.uniform(-s1,s1,[self.n_hidden_units,1]))
        self.W2=np.matrix(np.random.uniform(-s2,s2,[1,self.n_hidden_units]))
        self.b2=np.matrix(np.random.uniform(-s2,s2,[1,1]))
        
        #momentum initialization
        self.tW1=self.W1*0
        self.tb1=self.b1*0
        self.tW2=self.W2*0
        self.tb2=self.b2*0

        #the trainging process
        for i in range(self.n_epochs):
            for j in range(n_obs//self.batch_size):
                        self.X=train[:,j*self.batch_size:(j+1)*self.batch_size]
                        self.true=target[:,j*self.batch_size:(j+1)*self.batch_size]
                        self.forward()
                        self.backward()
                        self.update_weights()

    def predict(self,test):
        self.X=np.matrix(test).T
        self.forward()
        return np.squeeze(np.asarray(self.pred))