import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix

import mfem.ser as mfem
from mfem.ser import intArray

class aniso_diff:
    '''    
    MFEM testbed matrix interface
    
    Attributes
    ----------
    

    Methods
    -------
    
    __new__(prob_refinement, kwargs) : This function is called upon object 
        creation and returns a data dictionary with the matrix and other 
        items (see function documentation for __new__)

     aniso_discretization(self, prob_refinement, epsilon, theta) : 
        This function uses PyAMG stencils to create a 2D rotated anisotropic 
        diffusion problem for epsilon and theta 

    '''

    
    def __new__(self, prob_refinement, **kwargs):
        '''
        Return a rotated anisotropic diffusion discretization with 2D Q1 elements

        Input
        -----
        prob_refinement : refinement of problem (0, 1, 2, ...) to generate
        
        kwargs : dictionary of MFEM parameters, must contain the following keys
        
          'epsilon' : float 
                    anisotropy coefficient
        
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
        '''
        
        
        ##
        # Retrieve discretization paramters
        try:
            epsilon = kwargs ['epsilon']
            theta = kwargs['theta']
        except:
            raise ValueError("Incorrect kwargs for aniso diff generator")    
        
        ##
        # Stencil and matrix
        from pyamg.gallery import stencil_grid
        from pyamg.gallery.diffusion import diffusion_stencil_2d
        n = 32 * (2**prob_refinement)
        data = {}
        stencil = diffusion_stencil_2d(type='FE', epsilon=epsilon, theta=theta)
        data['A'] = stencil_grid(stencil, (n,n), format='csr')
        data['B'] = np.ones((data['A'].shape[0],1))
        X,Y = np.meshgrid(np.linspace(0,1.0,n), np.linspace(0,1.0,n))
        data['vertices'] = np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))
        data['b'] = np.zeros((data['A'].shape[0],1))
        data['docstring'] = 'Rotated Q1 Anisotripy on Unit Box by angle of %1.3e'%theta + \
                            'and PDE %1.3e u_xx + u_yy = f'%epsilon

        return data
        
