import numpy as np

import scipy.io as sio

import tensorflow as tf
import tensorflow.keras.utils
import tensorflow.keras.backend as K

def generateTheta(L,endim):
    theta_=np.random.normal(size=(L,endim))
    for l in range(L):
        theta_[l,:]=theta_[l,:]/np.sqrt(np.sum(theta_[l,:]**2))
    return theta_

def oneDWassersteinV3(p,q):
    # ~10 Times faster than V1

    # W2=(tf.nn.top_k(tf.transpose(p),k=tf.shape(p)[0]).values-
    #     tf.nn.top_k(tf.transpose(q),k=tf.shape(q)[0]).values)**2

    # return K.mean(W2, axis=-1)

    psort=tf.sort(p,axis=0)
    qsort=tf.sort(q,axis=0)
    pqmin=tf.minimum(K.min(psort,axis=0),K.min(qsort,axis=0))
    psort=psort-pqmin
    qsort=qsort-pqmin
    
    n_p=tf.shape(p)[0]
    n_q=tf.shape(q)[0]
    
    pcum=tf.multiply(tf.cast(tf.maximum(n_p,n_q),dtype='float32'),tf.divide(tf.cumsum(psort),tf.cast(n_p,dtype='float32')))
    qcum=tf.multiply(tf.cast(tf.maximum(n_p,n_q),dtype='float32'),tf.divide(tf.cumsum(qsort),tf.cast(n_q,dtype='float32')))
    
    indp=tf.cast(tf.floor(tf.linspace(0.,tf.cast(n_p,dtype='float32')-1.,tf.minimum(n_p,n_q)+1)),dtype='int32')
    indq=tf.cast(tf.floor(tf.linspace(0.,tf.cast(n_q,dtype='float32')-1.,tf.minimum(n_p,n_q)+1)),dtype='int32')
    
    phat=tf.gather(pcum,indp[1:],axis=0)
    phat=K.concatenate((K.expand_dims(phat[0,:],0),phat[1:,:]-phat[:-1,:]),0)
    
    qhat=tf.gather(qcum,indq[1:],axis=0)
    qhat=K.concatenate((K.expand_dims(qhat[0,:],0),qhat[1:,:]-qhat[:-1,:]),0)
          
    W2=K.mean((phat-qhat)**2,axis=0)
    return W2


def sWasserstein_hd(P,Q,theta,nclass,Cp=None,Cq=None):
    # High dimensional variant of the sWasserstein function

    '''
        P, Q - representations in embedding space between target & source
        theta - random matrix of directions
    '''

    p=K.dot(K.reshape(P, (-1, nclass)), K.transpose(theta))
    q=K.dot(K.reshape(Q, (-1, nclass)), K.transpose(theta))
    sw=K.mean(oneDWassersteinV3(p,q))

    return sw

def sWasserstein(P,Q,theta,nclass,Cp=None,Cq=None):
    '''
        P, Q - representations in embedding space between target & source
        theta - random matrix of directions
    '''
    p=K.dot(P,K.transpose(theta))
    q=K.dot(Q,K.transpose(theta))
    sw=K.mean(oneDWassersteinV3(p,q))

    return sw



    
def reinitLayers(model):
    # This code reinitialize a keras/tf model
    session = K.get_session()
    for layer in model.layers: 
        if isinstance(layer, keras.engine.topology.Container):
            reinitLayers(layer)
            continue
        for v in layer.__dict__:
            v_arg = getattr(layer,v)
            if hasattr(v_arg,'initializer'):
                initializer_method = getattr(v_arg, 'initializer')
                initializer_method.run(session=session)

 
def randperm(X,y):
    assert X.shape[0]==y.shape[0]
    ind=np.random.permutation(X.shape[0])
    X=X[ind,...]
    y=y[ind,...]
    return X,y

def batchGenerator(label,batchsize,nofclasses=2,seed=1,noflabeledsamples=None):
    N=label.shape[0]
    if not(noflabeledsamples):
        M=int(batchsize/nofclasses)
        ind=[]
        for i in range(nofclasses):
            labelIndex=np.argwhere(label[:,i]).squeeze()
            randInd=np.random.permutation(labelIndex.shape[0])
            ind.append(labelIndex[randInd[:M]])
        ind=np.asarray(ind).reshape(-1)
         
        labelout=label[ind]
    else:
        np.random.seed(seed)
        portionlabeled=min(batchsize/2,noflabeledsamples*nofclasses)
        M=portionlabeled/nofclasses
        indsupervised=[]
        indunsupervised=np.array([])
        for i in range(nofclasses):
            labelIndex=np.argwhere(label[:,i]).squeeze()
            randInd=np.random.permutation(labelIndex.shape[0])
            indsupervised.append(labelIndex[randInd[:noflabeledsamples]])
            indunsupervised=np.append(indunsupervised,np.array(labelIndex[randInd[noflabeledsamples:]]))
        np.random.seed()
        ind=[]  
        for i in range(nofclasses):
            ind.append(np.random.permutation(indsupervised[i])[:M])
        ind=np.asarray(ind).reshape(-1)
        indunsupervised=np.random.permutation(indunsupervised)      
        
        labelout=np.zeros((nofclasses*(batchsize/nofclasses),nofclasses))
        labelout[:portionlabeled]=label[ind,:]
        ind=np.concatenate((ind,indunsupervised[:nofclasses*(batchsize/nofclasses)-ind.shape[0]]))
    return ind.astype(int),labelout


