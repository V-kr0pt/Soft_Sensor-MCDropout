import numpy as np
from tensorflow import keras 
from scipy.special import logsumexp
from normalization import * 

class net:

  def __init__(self, tau, p_dropout, l, N, neurons):
    '''
    Inputs:
      tau -> precisão do modelo (float)

      p_dropout -> probabilidade de dropout (float)

      l -> length scale (float)

      neurons -> neuronios em cada camada (lista)
    '''

    self.tau = tau
    self.p_dropout = p_dropout
    self.l = l
    self.N = N
    self.neurons = neurons
        

    #Peso da regularização L2
    self.lbd =  self.l**2 *(1-self.p_dropout) / (2*self.N*self.tau)
    l2 = keras.regularizers.L2(self.lbd)

    #modelo Sequencial
    self.model = keras.models.Sequential()

    #rede
    for i,n in enumerate(self.neurons):
      if i == 0: 
        #entrada
        self.model.add(keras.layers.Input(shape=(self.neurons[i],))) 
        self.model.add(keras.layers.Dropout(p_dropout)) 

      else:
        if(i == len(self.neurons)-1):
          #saída
          self.model.add(keras.layers.Dense(self.neurons[i], activation=None, kernel_regularizer=l2)) 
        
        else:
          #camada escondida
          self.model.add(keras.layers.Dense(self.neurons[i], activation='tanh', kernel_regularizer=l2)) 
          self.model.add(keras.layers.Dropout(p_dropout))
    


  def MC_train(self, Xtn, ytn, epoch, lf='mean_squared_error', op='Adam'):
    '''
    Inputs:
      Xtn -> dados de entrada de treino normalizados (array)

      ytn -> dados de saída de treino normalizados (array)

      epoch -> qtd de épocas (int)

      lf -> loss function (str) | lf = 'mean_squared_error'

      op -> optimizer (op) | op = 'Adam'
    '''

    #Treinamento da rede
    print("-Treinando... -")
    self.model.compile(loss= lf, optimizer= op)
    self.model.fit(Xtn, ytn, epochs=epoch, verbose=0)
    print("-Treinamento Finalizado-")
  
  
  def MC_predict(self, Xtsn, ytsn, yts, T):
    '''
    Inputs:
      Xtsn -> dados de entrada de teste normalizados 

      ytsn -> dados de saída de teste normalizados 
      
      yts -> dados de saísa de teste não normalizados 
      
      T -> qtd de predições 

    Outputs:
      MC_pred -> predição da rede neural 

      MC_inc -> incerteza associada a predição 

      ll -> log-likelihood do modelo 
    '''

    #realizando T predições
    y_hat = np.array([self.model(Xtsn, training=True) for _ in range(T)])
    
    #Calculando RMSE
    MSE = ((y_hat - ytsn)**2).sum()/ytsn.shape[0]
    RMSE = np.sqrt(MSE)

    #Calculando o log-verossimilhança
    ll = (logsumexp(-0.5 * self.tau * (ytsn - y_hat)**2., 0) - np.log(T) 
            - 0.5*np.log(2*np.pi) + 0.5*np.log(self.tau))
    ll = np.mean(ll)

    #Calculando a média e a variância das predições
    mcpred = y_hat.mean(axis=0)
    mcvar = y_hat.var(axis=0)

    #Desnormalizando
    mcpred = Denormalize(mcpred, yts)
    mcvar = Denormalize(mcvar, yts)

    #Calculando a incerteza do modelo
    mcinc = self.tau**-1 + mcvar
    mcinc = np.sqrt(mcinc)
    
    self.RMSE = RMSE
    self.ll = ll
    self.pred = mcpred
    self.inc = mcinc
