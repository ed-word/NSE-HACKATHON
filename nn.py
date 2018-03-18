import numpy as np

class NeuralNetwork:
    
    def __init__(self, inp_n, hid_n, lr=0.001, momentum=0.9):
        self.n_input = inp_n
        self.n_hidden = hid_n
        self.learning_rate = lr
        self.momentum = momentum

        self.WLE1 = {
            'weight': np.random.normal(scale=0.1, size=[self.n_input, self.n_hidden])
        }

        self.WLE2 = {
            'weight': np.random.normal(scale=0.1, size=[self.n_hidden, 1])
        }

        self.WLD2 = {
            'weight': np.random.normal(scale=0.1, size=[1, self.n_hidden])
        }

        self.WLD1 = {
            'weight': np.random.normal(scale=0.1, size=[self.n_hidden, self.n_input])
        }
        
        activation_f = {
            'identity': lambda x: x,
            'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-x)),
            'tanh': lambda x: np.tanh(x),
            'relu': lambda x: x * (x > 0),
        }

        activation_f_prime = {
            'identity': lambda x: 1,
            'sigmoid': lambda x: x * (1.0 - x),
            'tanh': lambda x: 1 - x**2,
            'relu': lambda x: 1.0 * (x > 0),
        }
        
        activations = ['sigmoid', 'sigmoid']

        self.act_f1 = activation_f[activations[0]]
        self.act_f2 = activation_f[activations[1]]

        self.act_f1_prime = activation_f_prime[activations[0]]
        self.act_f2_prime = activation_f_prime[activations[1]]

        
        
    def __train(self, X, Y):
        if X.ndim < 2:
            X = np.expand_dims(X, axis=0)
        if Y.ndim < 2:
            Y = np.expand_dims(Y, axis=0)
            
        # forward prop
        h1E_inter = np.dot(X, self.WLE1['weight'])
        h1E_result = self.act_f1(h1E_inter)
        h2E_inter = np.dot(h1E_result, self.WLE2['weight'])
        h2E_result = self.act_f2(h2E_inter)
        
        h2D_inter = np.dot(h2E_result, self.WLD2['weight'])
        h2D_result = self.act_f2(h2D_inter)
        h1D_inter = np.dot(h2D_result, self.WLD1['weight'])
        h1D_result = self.act_f1(h1D_inter)
        
        # backward prop - 1
        error1 = np.mean(0.5 * np.square(h2E_result - Y))
        
        del_WLE2 = -np.multiply(Y - h2E_result, self.act_f2_prime(h2E_result))
        grad_WLE2 = np.dot(h1E_result.T, del_WLE2)
        
        del_WLE1 = np.dot(del_WLE2, self.WLE2['weight'].T) * self.act_f1_prime(h1E_result)
        grad_WLE1 = np.dot(X.T, del_WLE1)
        
        # backward prop - 2
        error2 = np.mean(0.5 * np.square(h1D_result - X))
        
        del_WLD1 = -np.multiply(X - h1D_result, self.act_f1_prime(h1D_result))
        grad_WLD1 = np.dot(h2D_result.T, del_WLD1)
        
        del_WLD2 = np.dot(del_WLD1, self.WLD1['weight'].T) * self.act_f2_prime(h2D_result)
        grad_WLD2 = np.dot(h2E_result.T, del_WLD2)
        
        return error1, error2, grad_WLE1, grad_WLE2, grad_WLD2, grad_WLD1
    
    
    def fit1(self, X, Y):
            
        e1, e2, grad_WLE1, grad_WLE2, grad_WLD2, grad_WLD1 = self.__train(X, Y)

        self.WLE1['weight'] -= self.learning_rate * grad_WLE1 + self.momentum * grad_WLE1
        self.WLE2['weight'] -= self.learning_rate * grad_WLE2 + self.momentum * grad_WLE2
        self.WLD2['weight'] -= self.learning_rate * grad_WLD2 + self.momentum * grad_WLD2
        self.WLD1['weight'] -= self.learning_rate * grad_WLD1 + self.momentum * grad_WLD1
        
        return e1, e2
    
    
    def predict(self):
        X = np.array([[1.0]])
        
        h2D_inter = np.dot(X, self.WLD2['weight'])
        h2D_result = self.act_f2(h2D_inter)
        h1D_inter = np.dot(h2D_result, self.WLD1['weight'])
        pred = self.act_f1(h1D_inter)
        
        return pred
     