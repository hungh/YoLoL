"""
Implementation of Adam Optimizer
"""
import numpy as np

class Adam:
    # global constant for Adam  optimizer
    MB_T = 'mb_iteration' # the iteration number of the mini-batch
    
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Args:
            beta1 (float): Beta 1 for Adam optimizer
            beta2 (float): Beta 2 for Adam optimizer            
            epsilon (float): Epsilon for Adam optimizer (prevents division by zero)
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    

    def update_parameters_once(self, parameters, grads, layer_index: int, learning_rate: float):
        mb_t = parameters[Adam.MB_T]        
        assert mb_t is not None, "mb_t must be initialized"

        # dW and db are calculated in the backward propagation (gradient descent) of the mini-batch
        dw = grads['dW' + str(layer_index)]

        parameters['VdW' + str(layer_index)] = self.beta1 * parameters['VdW' + str(layer_index)] + (1 - self.beta1) * dw
        parameters['SdW' + str(layer_index)] = self.beta2 * parameters['SdW' + str(layer_index)] + (1 - self.beta2) * (dw * dw)

        # Bias correction for weights
        parameters['VdW' + str(layer_index)] = parameters['VdW' + str(layer_index)] / (1 - self.beta1 ** mb_t)
        parameters['SdW' + str(layer_index)] = parameters['SdW' + str(layer_index)] / (1 - self.beta2 ** mb_t)

        # update Weights
        parameters['W' + str(layer_index)] = parameters['W' + str(layer_index)] - learning_rate * parameters['VdW' + str(layer_index)] / (np.sqrt(parameters['SdW' + str(layer_index)]) + self.epsilon)

        # handle batch norm specifics

        if '_batch_norm' in parameters:
            # Update gamma and beta
            dG = grads['dG' + str(layer_index)]
            dB = grads['dB' + str(layer_index)]

            # Update VdG, SdG, VdB, SdB
            parameters['VdG' + str(layer_index)] = self.beta1 * parameters['VdG' + str(layer_index)] + (1 - self.beta1) * dG
            parameters['SdG' + str(layer_index)] = self.beta2 * parameters['SdG' + str(layer_index)] + (1 - self.beta2) * (dG * dG)
            parameters['VdB' + str(layer_index)] = self.beta1 * parameters['VdB' + str(layer_index)] + (1 - self.beta1) * dB
            parameters['SdB' + str(layer_index)] = self.beta2 * parameters['SdB' + str(layer_index)] + (1 - self.beta2) * (dB * dB)

            # Bias correction
            VdG_corrected = parameters['VdG' + str(layer_index)] / (1 - self.beta1 ** mb_t)
            SdG_corrected = parameters['SdG' + str(layer_index)] / (1 - self.beta2 ** mb_t)
            VdB_corrected = parameters['VdB' + str(layer_index)] / (1 - self.beta1 ** mb_t)
            SdB_corrected = parameters['SdB' + str(layer_index)] / (1 - self.beta2 ** mb_t)

            # update gamma and beta
            parameters['G' + str(layer_index)] = parameters['G' + str(layer_index)] - learning_rate * VdG_corrected / (np.sqrt(SdG_corrected) + self.epsilon)
            parameters['B' + str(layer_index)] = parameters['B' + str(layer_index)] - learning_rate * VdB_corrected / (np.sqrt(SdB_corrected) + self.epsilon)
        else:
            # update bias
            db = grads['db' + str(layer_index)]
            parameters['Vdb' + str(layer_index)] = self.beta1 * parameters['Vdb' + str(layer_index)] + (1 - self.beta1) * db
            parameters['Sdb' + str(layer_index)] = self.beta2 * parameters['Sdb' + str(layer_index)] + (1 - self.beta2) * (db * db)

            # bias correction
            Vdb_corrected = parameters['Vdb' + str(layer_index)] / (1 - self.beta1 ** mb_t)
            Sdb_corrected = parameters['Sdb' + str(layer_index)] / (1 - self.beta2 ** mb_t)

            # update bias
            parameters['b' + str(layer_index)] = parameters['b' + str(layer_index)] - learning_rate * Vdb_corrected / (np.sqrt(Sdb_corrected) + self.epsilon)
               

    
    def initialize_parameters(self, parameters, layer_dims):
        """
        NOTE: layer_dims must include the input layer (X or A[0])
        """
        # initialize parameters for batch norm (if exists)
        if '_batch_norm' in parameters:
            for l in range(1, len(layer_dims)):
                parameters['VdW' + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
                parameters['SdW' + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
                parameters['VdG' + str(l)] = np.zeros((layer_dims[l], 1))
                parameters['SdG' + str(l)] = np.zeros((layer_dims[l], 1))
                # NOTE: batch norm does not have bias, 'B' is batch norm's parameter beta (matrix), bias should have been lower case 'b'
                parameters['VdB' + str(l)] = np.zeros((layer_dims[l], 1))
                parameters['SdB' + str(l)] = np.zeros((layer_dims[l], 1))
        else:
            for i in range(1, len(layer_dims)):
                parameters['VdW' + str(i)] = np.zeros((layer_dims[i], layer_dims[i-1]))
                parameters['SdW' + str(i)] = np.zeros((layer_dims[i], layer_dims[i-1]))
                parameters['Vdb' + str(i)] = np.zeros((layer_dims[i], 1))
                parameters['Sdb' + str(i)] = np.zeros((layer_dims[i], 1))


    def __str__(self):
        return f"Adam(beta1={self.beta1}, beta2={self.beta2}, epsilon={self.epsilon})"

