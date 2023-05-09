import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix

import mfem.ser as mfem
from mfem.ser import intArray

class coeffjump2D:
    '''
    Jumping coefficient diffusion problem (box in a box), 5-point Poisson testbed matrix interface
    
    Attributes
    ----------
    

    Methods
    -------
    
    __new__(prob_refinement, kwargs) : This function is called upon object 
        creation and returns a data dictionary with the matrix and other 
        items (see function documentation for __new__)

     coeffjump2D(self, prob_refinement, sigma) : 
        This function uses PyAMG stencils to create a classic jumping coefficient 
        problem of a box inside a box using 5-point finite-difference stencil.
        The inside box has a larger diffusion coefficient (sigma). 

    '''

    
    def __new__(self, prob_refinement, **kwargs):
        '''
        Return a jumping coefficient diffusion problem (box in a box) using 5-point FD

        Input
        -----
        prob_refinement : refinement of problem (0, 1, 2, ...) to generate
        
        kwargs : dictionary of parameters, must contain the following keys
        
          'sigma' : float
                       size of jump for inside box, suggested value is 1e3
        
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
            sigma = kwargs['sigma']
        except:
            raise ValueError("Incorrect kwargs for box-in-box coefficient jump generator")    
        
        ##
        # Stencil and matrix
        data = {}
        A, vertices, docstring = self.box_jump(self, prob_refinement, sigma)
        data['A'] = A
        data['B'] = np.ones((data['A'].shape[0],1))
        data['b'] = np.zeros((data['A'].shape[0],1))
        data['docstring'] = docstring
        data['vertices'] = vertices
        
        return data
        

    def box_jump(self, prob_refinement, sigma):
        '''
        Description
        ------------
        This function computes a matrix for the box-in-box coefficient jump of
        size sigma 
    
        Input
        ------
        prob_refinement : refinement of problem (0, 1, 2, ...) to generate
        sigma : Coefficient jump for inside region
    
        Output
        ------
        A : CSR matrix for discretization using 5-point FD
        vertices : array for regular grid of unit box
        docstring : string describing problem
        '''
        from scipy import sparse
        from pyamg import gallery

        # Define coefficient jump amount
        sigma1 = 1./3.
        sigma2 = sigma1*sigma
       
        # Generate a regular grid that has a multiple of 25 points in each direction
        # This way the interface lies on grid lines.
        # n = prob_refinement*25 + 26
        n = 25*(2**prob_refinement - 1) + 26
        wiggle = 1e-8
        X,Y = np.meshgrid(np.linspace(0,1.0,n), np.linspace(0,1.0,n))
        vertices = np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))
        
        # Find points interior to regions 1 and 2
        region1 = np.array(vertices[:,0] < 0.44-wiggle, dtype=int) + \
                  np.array(vertices[:,0] > 0.52+wiggle, dtype=int) 
        region11= np.array(vertices[:,1] < 0.44-wiggle, dtype=int) + \
                  np.array(vertices[:,1] > 0.52+wiggle, dtype=int)
        region1 = (region1 + region11).nonzero()[0]
        region2 = np.array(vertices[:,0] > 0.44+wiggle, dtype=int) + \
                  np.array(vertices[:,0] < 0.52-wiggle, dtype=int) + \
                  np.array(vertices[:,1] > 0.44+wiggle, dtype=int) + \
                  np.array(vertices[:,1] < 0.52-wiggle, dtype=int)
        region2 = (region2==4).nonzero()[0]
      
        # Find the N, S, E, W boundaries
        N = np.array(vertices[:,0] > 0.44+wiggle, dtype=int) + \
            np.array(vertices[:,0] < 0.52-wiggle, dtype=int) + \
            np.array( np.abs(vertices[:,1] - 0.52) < wiggle , dtype=int)
        N = (N==3).nonzero()[0]
        S = np.array(vertices[:,0] > 0.44+wiggle, dtype=int) + \
            np.array(vertices[:,0] < 0.52-wiggle, dtype=int) + \
            np.array( np.abs(vertices[:,1] - 0.44) < wiggle, dtype=int)
        S = (S==3).nonzero()[0]
        E = np.array(vertices[:,1] > 0.44+wiggle, dtype=int) + \
            np.array(vertices[:,1] < 0.52-wiggle, dtype=int) + \
            np.array( np.abs(vertices[:,0] - 0.52) < wiggle, dtype=int)
        E = (E==3).nonzero()[0]
        W = np.array(vertices[:,1] > 0.44+wiggle, dtype=int) + \
            np.array(vertices[:,1] < 0.52-wiggle, dtype=int) + \
            np.array( np.abs(vertices[:,0] - 0.44) < wiggle, dtype=int)
        W = (W==3).nonzero()[0]
        
        # Generate the N,S,E,W stencils in COO format
        N_col = np.hstack( (N, N+n, N-n, N+1, N-1) )
        N_row = np.hstack( (N, N,   N,   N,   N) )
        N_data = np.hstack(( (2*sigma1 + 2*sigma2)*np.ones((N.shape[0],)),    # Diagonal
                                           -sigma1*np.ones((N.shape[0],)),    # North Stencil Entry 
                                           -sigma2*np.ones((N.shape[0],)),    # South Stencil Entry 
                            -0.5*(sigma1 + sigma2)*np.ones((N.shape[0],)),    # East Stencil Entry 
                            -0.5*(sigma1 + sigma2)*np.ones((N.shape[0],)) ))  # West Stencil Entry 
        S_col = np.hstack( (S, S+n, S-n, S+1, S-1) )
        S_row = np.hstack( (S, S,   S,   S,   S) )
        S_data = np.hstack(( (2*sigma1 + 2*sigma2)*np.ones((S.shape[0],)),    # Diagonal
                                           -sigma2*np.ones((S.shape[0],)),    # North Stencil Entry 
                                           -sigma1*np.ones((S.shape[0],)),    # South Stencil Entry 
                            -0.5*(sigma1 + sigma2)*np.ones((S.shape[0],)),    # East Stencil Entry 
                            -0.5*(sigma1 + sigma2)*np.ones((S.shape[0],)) ))  # West Stencil Entry 
        E_col = np.hstack( (E, E+n, E-n, E+1, E-1) )
        E_row = np.hstack( (E, E,   E,   E,   E) )
        E_data = np.hstack(( (2*sigma1 + 2*sigma2)*np.ones((E.shape[0],)),    # Diagonal
                            -0.5*(sigma1 + sigma2)*np.ones((E.shape[0],)),    # North Stencil Entry 
                            -0.5*(sigma1 + sigma2)*np.ones((E.shape[0],)),    # South Stencil Entry 
                                           -sigma1*np.ones((E.shape[0],)),    # East Stencil Entry 
                                           -sigma2*np.ones((E.shape[0],)) ))  # West Stencil Entry 
        W_col = np.hstack( (W, W+n, W-n, W+1, W-1) )
        W_row = np.hstack( (W, W,   W,   W,   W) )
        W_data = np.hstack(( (2*sigma1 + 2*sigma2)*np.ones((W.shape[0],)),    # Diagonal
                            -0.5*(sigma1 + sigma2)*np.ones((W.shape[0],)),    # North Stencil Entry 
                            -0.5*(sigma1 + sigma2)*np.ones((W.shape[0],)),    # South Stencil Entry 
                                           -sigma2*np.ones((W.shape[0],)),    # East Stencil Entry 
                                           -sigma1*np.ones((W.shape[0],)) ))  # West Stencil Entry 
    
        # Setup boundary interface matrix
        A_boundary = sparse.coo_matrix( (N_data,(N_row,N_col)), shape=(n**2,n**2)) + \
                     sparse.coo_matrix( (S_data,(S_row,S_col)), shape=(n**2,n**2)) + \
                     sparse.coo_matrix( (E_data,(E_row,E_col)), shape=(n**2,n**2)) + \
                     sparse.coo_matrix( (W_data,(W_row,W_col)), shape=(n**2,n**2)) 
    
        # Find the four corners
        NW_corner = np.array( np.abs(vertices[:,0] - 0.44) < wiggle, dtype=int) + \
                    np.array( np.abs(vertices[:,1] - 0.52) < wiggle, dtype=int)
        NW_corner = (NW_corner==2).nonzero()[0]
        NE_corner = np.array( np.abs(vertices[:,0] - 0.52) < wiggle, dtype=int) + \
                    np.array( np.abs(vertices[:,1] - 0.52) < wiggle, dtype=int)
        NE_corner = (NE_corner==2).nonzero()[0]
        SW_corner = np.array( np.abs(vertices[:,0] - 0.44) < wiggle, dtype=int) + \
                    np.array( np.abs(vertices[:,1] - 0.44) < wiggle, dtype=int)
        SW_corner = (SW_corner==2).nonzero()[0]
        SE_corner = np.array( np.abs(vertices[:,0] - 0.52) < wiggle, dtype=int) + \
                    np.array( np.abs(vertices[:,1] - 0.44) < wiggle, dtype=int)
        SE_corner = (SE_corner==2).nonzero()[0]
        
        # Setup Corner Stencil Matrix
        #    Stencil entries for:   Diagaonl                      N                          S                    E                      W
        corner_col = np.array( (NW_corner,                NW_corner+n,              NW_corner-n,          NW_corner+1,            NW_corner-1,
                                SW_corner,                SW_corner+n,              SW_corner-n,          SW_corner+1,            SW_corner-1,
                                NE_corner,                NE_corner+n,              NE_corner-n,          NE_corner+1,            NE_corner-1,
                                SE_corner,                SE_corner+n,              SE_corner-n,          SE_corner+1,            SE_corner-1), dtype=int).reshape(-1,)
        corner_row = np.array( (NW_corner,                NW_corner  ,              NW_corner  ,          NW_corner  ,            NW_corner  ,
                                SW_corner,                SW_corner  ,              SW_corner  ,          SW_corner  ,            SW_corner  ,
                                NE_corner,                NE_corner  ,              NE_corner  ,          NE_corner  ,            NE_corner  ,
                                SE_corner,                SE_corner  ,              SE_corner  ,          SE_corner  ,            SE_corner  ), dtype=int).reshape(-1,)
        corner_data = np.array((3*sigma1 +   sigma2,-       sigma1,         -0.5*(sigma1 + sigma2),  -0.5*(sigma1 + sigma2), -       sigma1        ,
                                3*sigma1 +   sigma2,-0.5*(sigma1 + sigma2), -       sigma1        ,  -0.5*(sigma1 + sigma2), -       sigma1        ,
                                3*sigma1 +   sigma2,-       sigma1,         -0.5*(sigma1 + sigma2),  -       sigma1        , -0.5*(sigma1 + sigma2),
                                3*sigma1 +   sigma2,-0.5*(sigma1 + sigma2), -       sigma1        ,  -       sigma1        , -0.5*(sigma1 + sigma2),))
        A_corners = sparse.coo_matrix( (corner_data,(corner_row,corner_col)), shape=(n**2,n**2)) 
    
        # Generate constant stencil part of A for the purely interior regions 
        A = gallery.poisson((n,n), format='csr')
        I = sparse.eye(A.shape[0], A.shape[1], format='csr')
        I1 = I.copy()
        I1.data[:] = 0.0
        I1.data[region1] = 1.0
        I1.eliminate_zeros()
        I2 = I.copy()
        I2.data[:] = 0.0
        I2.data[region2] = 1.0
        I2.eliminate_zeros()
        A_interior = sigma1*I1*A + sigma2*I2*A
    
        A = A_interior + A_boundary + A_corners 
        docstring = "Square Coefficient Jump Diffusion\nSigma1 = %1.2e, Sigma2 = %1.2e\n"%(sigma1, sigma2)+\
              "Sigma1 is the diffusion coefficient on the outer region of the unit box,\n"+\
              "and Sigma2 defines the coefficient for the inner region over x,y in\n"+\
              "[0.44,0.52]x[0.44x0.52]. The interface lies on grid lines. Five-point FD used."
        return A, vertices, docstring 





