import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix

import mfem.ser as mfem
from mfem.ser import intArray

class adv2D:
    '''    
    Upwind finite-difference implementation of advection, testbed matrix interface
    
    Attributes
    ----------
    

    Methods
    -------
    
    __new__(prob_refinement, kwargs) : This function is called upon object 
        creation and returns a data dictionary with the matrix and other 
        items (see function documentation for __new__)

     adv2D(self, prob_refinement, theta) : 
        This function uses PyAMG stencils to create a 2D rotated advection
        problem for angle theta, using upwind finite differences. See 
        pyamg.gallery.advection2d for underlying generator.

    '''

    
    def __new__(self, prob_refinement, **kwargs):
        '''
        Return a rotated advection in 2D using upwind finite differences

        Input
        -----
        prob_refinement : refinement of problem (0, 1, 2, ...) to generate
        
        kwargs : dictionary of parameters, must contain the following keys
        
          'theta' : float
                       angle of rotation 
        
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
        See pyamg.gallery.advection2d for underlying generator
        '''
        
        
        ##
        # Retrieve discretization paramters
        try:
            theta = kwargs['theta']
        except:
            raise ValueError("Incorrect kwargs for advection 2D generator. Dictionary kwargs must contain `theta`, \n"+\
                             "the rotation angle for advection.")
        
        ##
        # Stencil and matrix
        from pyamg.gallery import advection_2d
        data = {}
        n = 2**(5 + prob_refinement)
        A,b = advection_2d((n,n), theta=theta)
        data['A'] = A
        data['B'] = np.ones((data['A'].shape[0],1))
        X,Y = np.meshgrid(np.linspace(0,1.0,n), np.linspace(0,1.0,n))
        data['vertices'] = np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))
        data['b'] = b
        data['docstring'] = '2D rotated advection problem for angle theta = %1.3e'%theta + \
                            ' using upwind finite differences and PDE (cos(theta),sin(theta)) dot grad(u) = 0.\n' +\
                            'Right-hand-side b contains boundary terms. ' +\
                            'See pyamg.gallery.advection2d for underlying generator.'

        return data
        
