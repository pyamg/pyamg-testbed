import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix

import mfem.ser as mfem
from mfem.ser import intArray

class poiss2D:
    '''    
    Classic 2D 5-point Poisson testbed matrix interface
    
    Attributes
    ----------
    

    Methods
    -------
    
    __new__(prob_refinement, kwargs) : This function is called upon object 
        creation and returns a data dictionary with the matrix and other 
        items (see function documentation for __new__)

     poiss2D(self, prob_refinement) : 
        This function uses PyAMG stencils to create a classic 5-point finite-difference
        discretization of the 2D Poisson operator. See pyamg.gallery.poisson for 
        underlying generator

    '''

    
    def __new__(self, prob_refinement, **kwargs):
        '''
        Return a 2D Poisson matrix with 5-point finite differences

        Input
        -----
        prob_refinement : refinement of problem (0, 1, 2, ...) to generate
        
        kwargs : dictionary of optional parameters, not currently used 
        
        Output
        ------
        Dictionary with the following keys
        
        A : csr matrix
            sparse discretization matrix
        
        b : array
            right-hand-side
        
        B : array
            near null space modes
        
        vertices : array
                 spatial vertices for each dof 
        
        docstring : string
                    describes discretization and problem parameters

        References
        ----------
        See pyamg.gallery.poisson for underlying generator
        '''

        ##
        # Stencil and matrix
        from pyamg import gallery
        data = {}
        n = 2**(5 + prob_refinement)
        data['A'] = gallery.poisson((n,n), format='csr')
        data['B'] = np.ones((data['A'].shape[0],1))
        X,Y = np.meshgrid(np.linspace(0,1.0,n), np.linspace(0,1.0,n))
        data['vertices'] = np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))
        data['b'] = np.zeros((data['A'].shape[0],1))
        data['docstring'] = '2D FD Poisson operator on Unit Box using 5-point stencil '+\
                            'See pyamg.gallery.poisson for underlying generator'

        return data
        