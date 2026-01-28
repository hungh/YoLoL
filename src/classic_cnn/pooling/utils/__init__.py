import numpy as np

class MaxPoolingUtils:
    @staticmethod
    def create_mask_from_window(x):
        """
        Creates a mask from an input matrix x, to identify the max entry of x.
        
        Arguments:
        x -- Array of shape (f, f)
        
        Returns:
        mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
        """    
        
        mask = x == np.max(x)
        
        return mask

        

class AveragePoolingUtils:
    @staticmethod
    def distribute_value(dz, shape):
        """
        Distributes the input value in the matrix of dimension shape
        
        Arguments:
        dz -- input scalar
        shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
        
        Returns:
        a -- Array of size (n_H, n_W) for which we distributed the value of dz
        """
        
        # Retrieve dimensions from shape (≈1 line)
        (n_H, n_W) = shape
        
        # Compute the value to distribute on the matrix (≈1 line)
        average = dz / (n_H * n_W)
        
        # Create a matrix where each entry is the "average" value (≈1 line)
        a = np.ones(shape) * average
        
        return a