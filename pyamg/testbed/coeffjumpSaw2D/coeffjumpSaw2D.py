import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix

import mfem.ser as mfem
from mfem.ser import intArray

class coeffjumpSaw2D:
    '''    
    Jumping coefficient diffusion problem (sawtooth jump), 5-point Poisson testbed matrix interface
    
    Attributes
    ----------
    

    Methods
    -------
    
    __new__(prob_refinement, kwargs) : This function is called upon object 
        creation and returns a data dictionary with the matrix and other 
        items (see function documentation for __new__)

     coeffjumpSaw2D(self, prob_refinement, sigma) :
        This function uses PyAMG stencils to create a classic jumping coefficient 
        problem where a sawtooth region has a larger diffusion coefficient (sigma). 
        Classic 5-point finite-differencing is used on [0, 16] x [0, 16]

    '''

    
    def __new__(self, prob_refinement, **kwargs):
        '''
        Return a jumping coefficient diffusion problem (sawtooth) using 5-point FD

        Input
        -----
        prob_refinement : refinement of problem (0, 1, 2, ...) to generate over domain
                          [0, 16] x [0, 16]
        
        kwargs : dictionary of parameters, must contain the following keys
        
          'sigma' : float
                       size of jump for sawtooth region, suggested value is 1e3
        
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
        "The multi-grid method for the diffusion equation with strongly
        discontinuous coefficients", RE Alcouffe, A Brandt, JE Dendy, Jr, JW
        Painter - SIAM Journal on Scientific, 1981
        '''
        
        
        ##
        # Retrieve discretization paramters
        try:
            sigma = kwargs['sigma']
        except:
            raise ValueError("Incorrect kwargs for sawtooth coefficient jump generator.  Dictionary kwargs must contain `sigma`, \n"+\
                             "the coefficient jump amount for the problem.")
 

        ##
        # Stencil and matrix
        data = {}
        A, vertices, docstring = self.sawtooth_jump(self, prob_refinement+1, sigma)
        data['A'] = A
        data['B'] = np.ones((data['A'].shape[0],1))
        data['b'] = np.zeros((data['A'].shape[0],1))
        data['docstring'] = docstring
        data['vertices'] = vertices

        return data

    def sawtooth_jump(self, prob_refinement, sigma):
        '''
        Description
        ------------
        This function computes a matrix for the sawtooth coefficient jump of
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
        from numpy import array, ravel, hstack, repeat, union1d, setdiff1d, meshgrid, linspace, arange
        from scipy.sparse import coo_matrix

        sigma1 = 1.0
        sigma2 = sigma1*sigma
        sigavg = 0.5*(sigma1 + sigma2)
            
        # Generate a regular grid that has a multiple of 17 points in each direction
        # This way the interface lies on grid lines.
        #n = h*16 + 17
        n = 16*(2**prob_refinement - 1) + 17
        wiggle = 1e-8
        X,Y = meshgrid(linspace(0,16.0,n), linspace(0,16.0,n))
        X = X.reshape(-1,1)
        Y = Y.reshape(-1,1)
        vertices = hstack((X, Y))
    
        # define member function that finds all points interior to the sawtooth
        def is_sawtooth_interior(x, y):
            # region 1
            interior = array( x > 1.0, dtype=int) + \
                       array( x < 3.0, dtype=int) + \
                       array( y < 5.0, dtype=int) + \
                       array( y > 0.0, dtype=int) 
            dof_list = (interior == 4).nonzero()[0]
    
            # region 3  (be careful with the >= and <=  !!)
            interior = array( x >= 3.0, dtype=int) + \
                       array( x <= 5.0, dtype=int) + \
                       array( y >  3.0, dtype=int) + \
                       array( y <  5.0, dtype=int)
            dof_list = hstack((dof_list, (interior == 4).nonzero()[0] ))
            
            # region 4
            interior = array( x > 5.0, dtype=int) + \
                       array( x < 7.0, dtype=int) + \
                       array( y > 3.0, dtype=int) + \
                       array( y < 7.0, dtype=int)
            dof_list = hstack((dof_list, (interior == 4).nonzero()[0] ))
        
            # region 5 (be careful with the >= and <=  !!)
            interior = array( x >= 7.0, dtype=int) + \
                       array( x <= 9.0, dtype=int) + \
                       array( y >  5.0, dtype=int) + \
                       array( y <  7.0, dtype=int)
            dof_list = hstack((dof_list, (interior == 4).nonzero()[0] ))
        
            # region 6
            interior = array( x > 9.0, dtype=int) + \
                       array( x < 11., dtype=int) + \
                       array( y > 5.0, dtype=int) + \
                       array( y < 9.0, dtype=int)
            dof_list = hstack((dof_list, (interior == 4).nonzero()[0] ))
        
            # region 7 (be careful with the >= and <=  !!)
            interior = array( x >= 11., dtype=int) + \
                       array( x <= 13., dtype=int) + \
                       array( y >  7.0, dtype=int) + \
                       array( y <  9.0, dtype=int)
            dof_list = hstack((dof_list, (interior == 4).nonzero()[0] ))
        
            # region 8
            interior = array( x > 13., dtype=int) + \
                       array( x < 15., dtype=int) + \
                       array( y > 7.0, dtype=int) + \
                       array( y < 13., dtype=int)
            dof_list = hstack((dof_list, (interior == 4).nonzero()[0] ))
        
            # region 9 (be careful with the >= and <=  !!)
            interior = array( x >= 15., dtype=int) + \
                       array( x <  16., dtype=int) + \
                       array( y >  11., dtype=int) + \
                       array( y <  13., dtype=int)
            dof_list = hstack((dof_list, (interior == 4).nonzero()[0] ))
            
            return dof_list
    
        # define member function that finds all East facing sawtooth walls
        #   we don't want any corner points...
        def is_sawtooth_interior_east_wall(x, y):
            # region 1
            interior = array( x == 1.0, dtype=int) + \
                       array( y > 0.0, dtype=int)  + \
                       array( y < 5.0, dtype=int)
            dof_list = (interior == 3).nonzero()[0]
    
            # region 4
            interior = array( x == 5.0, dtype=int) +\
                       array( y > 5.0, dtype=int)  +\
                       array( y < 7.0, dtype=int)
            dof_list = hstack((dof_list, (interior == 3).nonzero()[0] ))
        
            # region 7
            interior = array( x == 9., dtype=int) + \
                       array( y > 7.0, dtype=int) + \
                       array( y < 9.0, dtype=int)
            dof_list = hstack((dof_list, (interior == 3).nonzero()[0] ))
        
            # region 8
            interior = array( x == 13., dtype=int) + \
                       array( y > 9.0, dtype=int) +  \
                       array( y < 13., dtype=int)
            dof_list = hstack((dof_list, (interior == 3).nonzero()[0] ))
        
            return dof_list
    
        # define member function that finds all Southern sawtooth walls
        #   we don't want any corner points...
        def is_sawtooth_interior_south_wall(x, y):
            # region 1
            interior = array( x > 3.0, dtype=int) + \
                       array( x < 7.0, dtype=int) + \
                       array( y == 3.0, dtype=int)
            dof_list = (interior == 3).nonzero()[0]
    
            # region 4
            interior = array( x > 7.0, dtype=int) + \
                       array( x < 11.0, dtype=int) +\
                       array( y == 5.0, dtype=int)
            dof_list = hstack( (dof_list, (interior == 3).nonzero()[0] ))
    
            # region 7
            interior = array( x > 11.0, dtype=int) + \
                       array( x < 15.0, dtype=int) + \
                       array( y == 7.0, dtype=int)
            dof_list = hstack( (dof_list, (interior == 3).nonzero()[0] ))
        
            # region 8
            interior = array( x > 15.0, dtype=int) + \
                       array( x < 16.0, dtype=int) + \
                       array( y == 11.0, dtype=int)
            dof_list = hstack( (dof_list, (interior == 3).nonzero()[0] ))
    
            return dof_list
    
        # define member function that finds all Northern sawtooth walls
        #   we don't want any corner points...
        def is_sawtooth_interior_north_wall(x, y):
            # region 1
            interior = array( x > 1.0, dtype=int) + \
                       array( x < 5.0, dtype=int) + \
                       array( y == 5.0, dtype=int)
            dof_list = (interior == 3).nonzero()[0]
    
            # region 4
            interior = array( x > 5.0, dtype=int) + \
                       array( x < 9.0, dtype=int) + \
                       array( y == 7.0, dtype=int)
            dof_list = hstack((dof_list, (interior == 3).nonzero()[0] ))
    
            # region 7
            interior = array( x > 9.0, dtype=int) + \
                       array( x < 13.0, dtype=int) +\
                       array( y == 9.0, dtype=int)
            dof_list = hstack((dof_list, (interior == 3).nonzero()[0] ))
        
            # region 8
            interior = array( x > 13.0, dtype=int) + \
                       array( x < 16.0, dtype=int) + \
                       array( y == 13.0, dtype=int)
            dof_list = hstack((dof_list, (interior == 3).nonzero()[0] )) 
    
            return dof_list
    
        # define member function that finds all West facing sawtooth walls
        #   we don't want any corner points...
        def is_sawtooth_interior_west_wall(x, y):
            # region 1
            interior = array( x == 3.0, dtype=int) + \
                       array( y > 0.0, dtype=int) +  \
                       array( y < 3.0, dtype=int)
            dof_list = (interior == 3).nonzero()[0]
    
            # region 4
            interior = array( x == 7.0, dtype=int) + \
                       array( y > 3.0, dtype=int) +  \
                       array( y < 5.0, dtype=int)
            dof_list = hstack((dof_list, (interior == 3).nonzero()[0] ))
        
            # region 7
            interior = array( x == 11., dtype=int) + \
                       array( y > 5.0, dtype=int) +  \
                       array( y < 7.0, dtype=int)
            dof_list = hstack((dof_list, (interior == 3).nonzero()[0] ))
        
            # region 8
            interior = array( x == 15., dtype=int) + \
                       array( y > 7.0, dtype=int) +  \
                       array( y < 11., dtype=int)
            dof_list = hstack((dof_list, (interior == 3).nonzero()[0] ))
    
            return dof_list
    
        # define member function that finds all outer corners on the sawtooth top
        def is_sawtooth_interior_top_outer_corner(x, y):
            # region 1
            interior = array( x == 1.0, dtype=int) + \
                       array( y == 5.0, dtype=int)  
            dof_list = (interior == 2).nonzero()[0]
    
            # region 4
            interior = array( x == 5.0, dtype=int) + \
                       array( y == 7.0, dtype=int)
            dof_list = hstack((dof_list, (interior == 2).nonzero()[0] ))
        
            # region 7
            interior = array( x == 9., dtype=int) + \
                       array( y == 9.0, dtype=int)
            dof_list = hstack((dof_list, (interior == 2).nonzero()[0] ))
        
            # region 8
            interior = array( x == 13., dtype=int) + \
                       array( y == 13., dtype=int)
            dof_list = hstack((dof_list, (interior == 2).nonzero()[0] ))
        
            return dof_list
    
        # define member function that finds all inner corners on the sawtooth top
        def is_sawtooth_interior_top_inner_corner(x, y):
            # region 1
            interior = array( x == 5.0, dtype=int) + \
                       array( y == 5.0, dtype=int)  
            dof_list = (interior == 2).nonzero()[0]
    
            # region 4
            interior = array( x == 9.0, dtype=int) + \
                       array( y == 7.0, dtype=int)
            dof_list = hstack((dof_list, (interior == 2).nonzero()[0] ))
        
            # region 7
            interior = array( x == 13., dtype=int) + \
                       array( y == 9.0, dtype=int)
            dof_list = hstack((dof_list, (interior == 2).nonzero()[0] ))
    
            return dof_list
    
        # define member function that finds all inner corners on the sawtooth bottom
        def is_sawtooth_interior_bottom_inner_corner(x, y):
            # region 1
            interior = array( x == 3.0, dtype=int) + \
                       array( y == 3.0, dtype=int)  
            dof_list = (interior == 2).nonzero()[0]
    
            # region 4
            interior = array( x == 7.0, dtype=int) + \
                       array( y == 5.0, dtype=int)
            dof_list = hstack((dof_list, (interior == 2).nonzero()[0] ))
        
            # region 7
            interior = array( x == 11., dtype=int) + \
                       array( y == 7.0, dtype=int)
            dof_list = hstack((dof_list, (interior == 2).nonzero()[0] ))
        
            # region 8
            interior = array( x == 15., dtype=int) + \
                       array( y == 11., dtype=int)
            dof_list = hstack((dof_list, (interior == 2).nonzero()[0] ))
    
            return dof_list

        # define member function that finds all outer corners on the sawtooth bottom
        def is_sawtooth_interior_bottom_outer_corner(x, y):
            # region 1
            interior = array( x == 7.0, dtype=int) + \
                       array( y == 3.0, dtype=int)  
            dof_list = (interior == 2).nonzero()[0]
    
            # region 4
            interior = array( x == 11.0, dtype=int) + \
                       array( y == 5.0, dtype=int)
            dof_list = hstack((dof_list, (interior == 2).nonzero()[0] ))
        
            # region 7
            interior = array( x == 15., dtype=int) + \
                       array( y == 7.0, dtype=int)
            dof_list = hstack((dof_list, (interior == 2).nonzero()[0] ))
        
            return dof_list
    
        # define member function that finds the Southern sawtooth boundary
        def is_sawtooth_southern_wall(x, y):
            # region 1
            interior = array( x > 1.0, dtype=int) + \
                       array( x < 3.0, dtype=int) + \
                       array( y == 0.0, dtype=int)  
            dof_list = (interior == 3).nonzero()[0]
        
            return dof_list
    
        # define member function that finds the left Southern sawtooth boundary point
        def is_sawtooth_left_southern_boundary_point(x, y):
            # region 1
            interior = array( x == 1.0, dtype=int) + \
                       array( y == 0.0, dtype=int)  
            dof_list = (interior == 2).nonzero()[0]
        
            return dof_list
    
        # define member function that finds the right Southern sawtooth boundary point
        def is_sawtooth_right_southern_boundary_point(x, y):
            # region 1
            interior = array( x == 3.0, dtype=int) + \
                       array( y == 0.0, dtype=int)  
            dof_list = (interior == 2).nonzero()[0]
    
            return dof_list
    
        # define member function that finds the Western sawtooth boundary
        def is_sawtooth_western_wall(x, y):
            # region 1
            interior = array( x == 16., dtype=int) + \
                       array( y > 11., dtype=int)  + \
                       array( y < 13., dtype=int)  
            dof_list = (interior == 3).nonzero()[0]
    
            return dof_list
    
        # define member function that finds the top Western sawtooth boundary point
        def is_sawtooth_top_western_boundary_point(x, y):
            # region 1
            interior = array( x == 16., dtype=int) + \
                       array( y == 13., dtype=int)  
            dof_list = (interior == 2).nonzero()[0]
        
            return dof_list
    
        # define member function that finds the bottom Western sawtooth boundary point
        def is_sawtooth_bottom_western_boundary_point(x, y):
            # region 1
            interior = array( x == 16., dtype=int) + \
                       array( y == 11., dtype=int)  
            dof_list = (interior == 2).nonzero()[0]
    
            return dof_list
    
        # define member function that finds the Northern wall 
        # no corner points...
        def is_northern_wall(x, y):
            # region 1
            interior = array( y == 16., dtype=int)  + \
                       array( x  > 0. , dtype=int)  + \
                       array( x  < 16. , dtype=int)  
            dof_list = (interior == 3).nonzero()[0]
        
            return dof_list
    
        # define member function that finds the Southern wall 
        # no corner points...
        def is_southern_wall(x, y):
            # region 1
            interior = array( y == 0., dtype=int)   + \
                       array( x  > 0. , dtype=int)  + \
                       array( x  < 16. , dtype=int)  
            dof_list = (interior == 3).nonzero()[0]
      
            sawtooth_pts = union1d( is_sawtooth_southern_wall(x, y),
                union1d(is_sawtooth_left_southern_boundary_point(x, y),
                        is_sawtooth_right_southern_boundary_point(x, y) ))
            
            return setdiff1d( dof_list, sawtooth_pts ) 
     
        # define member function that finds the Eastern wall 
        # no corner points...
        def is_eastern_wall(x, y):
            # region 1
            interior = array( x == 0., dtype=int)   + \
                       array( y  > 0. , dtype=int)  + \
                       array( y  < 16. , dtype=int)  
            dof_list = (interior == 3).nonzero()[0]
    
            return dof_list
    
        # define member function that finds the Western wall 
        # no corner points...
        def is_western_wall(x, y):
            # region 1
            interior = array( x == 16., dtype=int)  + \
                       array( y  > 0. , dtype=int)  + \
                       array( y  < 16. , dtype=int)  
            dof_list = (interior == 3).nonzero()[0]
    
            sawtooth_pts = union1d( is_sawtooth_western_wall(x, y),
                union1d(is_sawtooth_top_western_boundary_point(x, y),
                        is_sawtooth_bottom_western_boundary_point(x, y) ))
            
            return setdiff1d( dof_list, sawtooth_pts ) 

        # define member function that finds the top right corner
        def is_top_right_corner(x, y):
            # region 1
            interior = array( x == 16., dtype=int) + \
                       array( y == 16., dtype=int)  
            dof_list = (interior == 2).nonzero()[0]
        
            return dof_list
    
        # define member function that finds the top left corner
        def is_top_left_corner(x, y):
            # region 1
            interior = array( x == 0., dtype=int) + \
                       array( y == 16., dtype=int)  
            dof_list = (interior == 2).nonzero()[0]
    
            return dof_list
    
        # define member function that finds the bottom right corner
        def is_bottom_right_corner(x, y):
            # region 1
            interior = array( x == 16., dtype=int) + \
                       array( y == 0., dtype=int)  
            dof_list = (interior == 2).nonzero()[0]
        
            return dof_list
    
        # define member function that finds the bottom left corner
        def is_bottom_left_corner(x, y):
            # region 1
            interior = array( x == 0., dtype=int) + \
                       array( y == 0., dtype=int)  
            dof_list = (interior == 2).nonzero()[0]
        
            return dof_list
    
        # define member function that finds all interior points not part of the
        # sawtooth
        def is_nonsawtooth_interior(x,y):
            sawtooth_pts = union1d( is_sawtooth_interior(x, y),
                union1d(is_sawtooth_interior_east_wall(x, y),
                union1d(is_sawtooth_interior_south_wall(x, y),
                union1d(is_sawtooth_interior_north_wall(x, y),
                union1d(is_sawtooth_interior_west_wall(x, y),
                union1d(is_sawtooth_interior_top_outer_corner(x, y),
                union1d(is_sawtooth_interior_top_inner_corner(x, y),
                union1d(is_sawtooth_interior_bottom_inner_corner(x, y),
                union1d(is_sawtooth_interior_bottom_outer_corner(x, y),
                union1d(is_sawtooth_southern_wall(x, y),
                union1d(is_sawtooth_left_southern_boundary_point(x, y),
                union1d(is_sawtooth_right_southern_boundary_point(x, y),
                union1d(is_sawtooth_western_wall(x, y),
                union1d(is_sawtooth_top_western_boundary_point(x, y),
                        is_sawtooth_bottom_western_boundary_point(x, y) ))))))))))))))
            bdy_pts = union1d( is_northern_wall(x, y),
                union1d(is_southern_wall(x, y),
                union1d(is_eastern_wall(x, y),
                union1d(is_western_wall(x, y),
                union1d(is_top_right_corner(x, y),
                union1d(is_top_left_corner(x, y),
                union1d(is_bottom_right_corner(x, y),
                        is_bottom_left_corner(x, y) )))))))
            return setdiff1d( arange(x.shape[0]), union1d(sawtooth_pts, bdy_pts) ) 

        ##
        # Create matrix
        A = coo_matrix( (n**2, n**2) )
        indys_count = 0

        ##
        # Put entries in matrix for each type of point
        
        #                                          diag            below    above     left     right
        indys     = is_sawtooth_interior_east_wall(X, Y)
        indys_count += indys.shape[0]
        row_indys = repeat(indys, 5)
        offset    = ravel( repeat( array([[          0,              -n,      +n,       -1,       +1]]), indys.shape[0], axis=0))
        stencils  = ravel( repeat( array([[ 2*sigma1 + 2*sigma2,  -sigavg, -sigavg, -sigma1, -sigma2]]), indys.shape[0], axis=0))
        col_indys = row_indys + offset
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
    
        #                                          diag            below    above     left     right
        indys     = is_sawtooth_interior_west_wall(X, Y)
        indys_count += indys.shape[0]
        row_indys = repeat(indys, 5)
        offset    = ravel( repeat( array([[          0,              -n,      +n,       -1,       +1]]), indys.shape[0], axis=0))
        stencils  = ravel( repeat( array([[ 2*sigma1 + 2*sigma2,  -sigavg, -sigavg, -sigma2, -sigma1]]), indys.shape[0], axis=0))
        col_indys = row_indys + offset
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
    
        #                                          diag            below    above     left     right
        indys     = is_sawtooth_interior_south_wall(X, Y)
        indys_count += indys.shape[0]
        row_indys = repeat(indys, 5)
        offset    = ravel( repeat( array([[          0,              -n,      +n,       -1,       +1]]), indys.shape[0], axis=0))
        stencils  = ravel( repeat( array([[ 2*sigma1 + 2*sigma2,  -sigma1, -sigma2, -sigavg, -sigavg]]), indys.shape[0], axis=0))
        col_indys = row_indys + offset
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
    
        #                                          diag            below    above     left     right
        indys     = is_sawtooth_interior_north_wall(X, Y)
        indys_count += indys.shape[0]
        row_indys = repeat(indys, 5)
        offset    = ravel( repeat( array([[          0,              -n,      +n,       -1,       +1]]), indys.shape[0], axis=0))
        stencils  = ravel( repeat( array([[ 2*sigma1 + 2*sigma2,  -sigma2, -sigma1, -sigavg, -sigavg]]), indys.shape[0], axis=0))
        col_indys = row_indys + offset
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
    
        #                                          diag            below    above     left     right
        indys     = is_sawtooth_interior_top_outer_corner(X, Y)
        indys_count += indys.shape[0]
        row_indys = repeat(indys, 5)
        offset    = ravel( repeat( array([[          0,              -n,      +n,       -1,       +1]]), indys.shape[0], axis=0))
        stencils  = ravel( repeat( array([[ 3*sigma1 + sigma2,   -sigavg, -sigma1, -sigma1, -sigavg]]), indys.shape[0], axis=0))
        col_indys = row_indys + offset
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
    
        #                                          diag            below    above     left     right
        indys     = is_sawtooth_interior_top_inner_corner(X, Y)
        indys_count += indys.shape[0]
        row_indys = repeat(indys, 5)
        offset    = ravel( repeat( array([[          0,              -n,      +n,       -1,       +1]]), indys.shape[0], axis=0))
        stencils  = ravel( repeat( array([[   sigma1 + 3*sigma2,  -sigma2, -sigavg, -sigavg, -sigma2]]), indys.shape[0], axis=0))
        col_indys = row_indys + offset
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
    
        #                                          diag            below    above     left     right
        indys     = is_sawtooth_interior_bottom_inner_corner(X, Y)
        indys_count += indys.shape[0]
        row_indys = repeat(indys, 5)
        offset    = ravel( repeat( array([[          0,              -n,      +n,       -1,       +1]]), indys.shape[0], axis=0))
        stencils  = ravel( repeat( array([[   sigma1 + 3*sigma2,  -sigavg, -sigma2, -sigma2, -sigavg]]), indys.shape[0], axis=0))
        col_indys = row_indys + offset
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
    
        #                                          diag            below    above     left     right
        indys     = is_sawtooth_interior_bottom_outer_corner(X, Y)
        indys_count += indys.shape[0]
        row_indys = repeat(indys, 5)
        offset    = ravel( repeat( array([[          0,              -n,      +n,       -1,       +1]]), indys.shape[0], axis=0))
        stencils  = ravel( repeat( array([[ 3*sigma1 +   sigma2,  -sigma1, -sigavg, -sigavg, -sigma1]]), indys.shape[0], axis=0))
        col_indys = row_indys + offset
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
        
        # nonzero rowsum here, it's a boundary point
        #                                          diag            below    above     left     right
        indys     = is_sawtooth_southern_wall(X, Y)
        indys_count += indys.shape[0]
        row_indys = repeat(indys, 4)
        offset    = ravel( repeat( array([[          0,              +n,       -1,       +1]]), indys.shape[0], axis=0))
        stencils  = ravel( repeat( array([[      4*sigma2,       -sigma2, -sigma2, -sigma2]]), indys.shape[0], axis=0))
        col_indys = row_indys + offset
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
    
        # nonzero rowsum here, it's a boundary point
        #                                          diag            below    above     left     right
        indys     = is_sawtooth_left_southern_boundary_point(X, Y)
        indys_count += indys.shape[0]
        row_indys = repeat(indys, 4)
        offset    = ravel( repeat( array([[          0,              +n,       -1,       +1]]), indys.shape[0], axis=0))
        stencils  = ravel( repeat( array([[      4*sigavg,       -sigavg, -sigma1, -sigma2]]), indys.shape[0], axis=0))
        col_indys = row_indys + offset
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
    
        # nonzero rowsum here, it's a boundary point
        #                                          diag            below    above     left     right
        indys     = is_sawtooth_right_southern_boundary_point(X, Y)
        indys_count += indys.shape[0]
        row_indys = repeat(indys, 4)
        offset    = ravel( repeat( array([[          0,              +n,       -1,       +1]]), indys.shape[0], axis=0))
        stencils  = ravel( repeat( array([[     4*sigavg,        -sigavg, -sigma2, -sigma1]]), indys.shape[0], axis=0))
        col_indys = row_indys + offset
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
    
        # nonzero rowsum here, it's a boundary point
        #                                          diag            below    above     left     right
        indys     = is_sawtooth_western_wall(X, Y)
        indys_count += indys.shape[0]
        row_indys = repeat(indys, 4)
        offset    = ravel( repeat( array([[          0,              -n,      +n,       -1]]), indys.shape[0], axis=0))
        stencils  = ravel( repeat( array([[      4*sigma2,       -sigma2, -sigma2, -sigma2]]), indys.shape[0], axis=0))
        col_indys = row_indys + offset
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
        
        # nonzero rowsum here, it's a boundary point
        #                                          diag            below    above     left     right
        indys     = is_sawtooth_top_western_boundary_point(X, Y)
        indys_count += indys.shape[0]
        row_indys = repeat(indys, 4)
        offset    = ravel( repeat( array([[          0,              -n,      +n,       -1]]), indys.shape[0], axis=0))
        stencils  = ravel( repeat( array([[      4*sigavg,       -sigma2, -sigma1, -sigavg]]), indys.shape[0], axis=0))
        col_indys = row_indys + offset
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
    
        # nonzero rowsum here, it's a boundary point
        #                                          diag            below    above     left     right
        indys     = is_sawtooth_bottom_western_boundary_point(X, Y)
        indys_count += indys.shape[0]
        row_indys = repeat(indys, 4)
        offset    = ravel( repeat( array([[          0,              -n,      +n,       -1]]), indys.shape[0], axis=0))
        stencils  = ravel( repeat( array([[      4*sigavg,       -sigma1, -sigma2, -sigavg]]), indys.shape[0], axis=0))
        col_indys = row_indys + offset
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
    
        indys     = is_northern_wall(X, Y)
        indys_count += indys.shape[0]
        offset    = ravel( repeat( array([[0, -n, -1, +1]]), indys.shape[0], axis=0))
        row_indys = repeat(indys, 4)
        col_indys = row_indys + offset
        stencils  = ravel( repeat( sigma1*array([[ 4., -1., -1., -1]]), indys.shape[0], axis=0))
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
        
        indys     = is_southern_wall(X, Y)
        indys_count += indys.shape[0]
        offset    = ravel( repeat( array([[0, +n, -1, +1]]), indys.shape[0], axis=0))
        row_indys = repeat(indys, 4)
        col_indys = row_indys + offset
        stencils  = ravel( repeat( sigma1*array([[ 4., -1., -1., -1]]), indys.shape[0], axis=0))
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
    
        indys     = is_eastern_wall(X, Y)
        indys_count += indys.shape[0]
        offset    = ravel( repeat( array([[0, -n, +n, +1]]), indys.shape[0], axis=0))
        row_indys = repeat(indys, 4)
        col_indys = row_indys + offset
        stencils  = ravel( repeat( sigma1*array([[ 4., -1., -1., -1]]), indys.shape[0], axis=0))
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
    
        indys     = is_western_wall(X, Y)
        indys_count += indys.shape[0]
        offset    = ravel( repeat( array([[0, -n, +n, -1]]), indys.shape[0], axis=0))
        row_indys = repeat(indys, 4)
        col_indys = row_indys + offset
        stencils  = ravel( repeat( sigma1*array([[ 4., -1., -1., -1]]), indys.shape[0], axis=0))
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
    
        indys     = is_top_right_corner(X, Y)
        indys_count += indys.shape[0]
        offset    = ravel( repeat( array([[0, -n, -1]]), indys.shape[0], axis=0))
        row_indys = repeat(indys, 3)
        col_indys = row_indys + offset
        stencils  = ravel( repeat( sigma1*array([[ 4., -1., -1.]]), indys.shape[0], axis=0))
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
    
        indys     = is_top_left_corner(X, Y)
        indys_count += indys.shape[0]
        offset    = ravel( repeat( array([[0, -n, +1]]), indys.shape[0], axis=0))
        row_indys = repeat(indys, 3)
        col_indys = row_indys + offset
        stencils  = ravel( repeat( sigma1*array([[ 4., -1., -1.]]), indys.shape[0], axis=0))
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
    
        indys     = is_bottom_right_corner(X, Y)
        indys_count += indys.shape[0]
        offset    = ravel( repeat( array([[0, +n, -1]]), indys.shape[0], axis=0))
        row_indys = repeat(indys, 3)
        col_indys = row_indys + offset
        stencils  = ravel( repeat( sigma1*array([[ 4., -1., -1.]]), indys.shape[0], axis=0))
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
    
        indys     = is_bottom_left_corner(X, Y)
        indys_count += indys.shape[0]
        offset    = ravel( repeat( array([[0, +n, +1]]), indys.shape[0], axis=0))
        row_indys = repeat(indys, 3)
        col_indys = row_indys + offset
        stencils  = ravel( repeat( sigma1*array([[ 4., -1., -1.]]), indys.shape[0], axis=0))
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
    
        indys     = is_nonsawtooth_interior(X, Y)
        indys_count += indys.shape[0]
        offset    = ravel( repeat( array([[0, -n, +n, -1, +1]]), indys.shape[0], axis=0))
        row_indys = repeat(indys, 5)
        col_indys = row_indys + offset
        stencils  = ravel( repeat( sigma1*array([[ 4., -1., -1., -1., -1.]]), indys.shape[0], axis=0))
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
        
        indys     = is_sawtooth_interior(X, Y)
        indys_count += indys.shape[0]
        offset    = ravel( repeat( array([[0, -n, +n, -1, +1]]), indys.shape[0], axis=0))
        row_indys = repeat(indys, 5)
        col_indys = row_indys + offset
        stencils  = ravel( repeat( sigma2*array([[ 4., -1., -1., -1., -1.]]), indys.shape[0], axis=0))
        A = A + coo_matrix( (stencils, (row_indys, col_indys)), shape=A.shape)
        
        docstring = "Sawtooth Coefficient Jump Diffusion\nSigma1 = %1.2e, Sigma2 = %1.2e\n"%(sigma1, sigma2)+\
              "Sigma1 is the diffusion coefficient on the outer region of the domain [0, 16] x [0, 16], \n"+\
              "and Sigma2 defines the coefficient for the inner sawtooth region. The interface\n"+\
              "lies on grid lines.  See Dendy and Alcouffe for details.  Use 5-point FD.\n"

        if False:
            # print some diagnostics for debugging
            A = A.tocsr()
            rowsum = abs(A*numpy.ones((A.shape[0],1)) )
            rowsum_max = rowsum.max()
            rowsum_indy = (abs(rowsum) > wiggle).nonzero()[0]
            zero_rows = ((A.indptr[1:] - A.indptr[:-1]) == 0).nonzero()[0]
            print("indys_count is              %d     (should be %d)"%(indys_count, A.shape[0]) )
            print("Max row sum is              %1.2e (should be 1.00e+03)"%rowsum_max )
            print("Numer of nonzero rowsums is %d      (should be %d)"%(rowsum_indy.shape[0], 4*n-4) )
            print("Zero rows at indices                 (should be empty)\n" + str(zero_rows) )
            unique_entries = unique(A.data)
            print("Num unique entries of A is  %d        (should be 8)"%unique_entries.shape[0])
            #print(" those entries are " + str(unique_entries)
            
            from pyamg.gallery import poisson
            Poiss = poisson( (n,n), format='csr')
            Acopy = A.copy()
            Acopy.data[:] = 1.0
            Poiss.data[:] = 1.0
            diff = Acopy - Poiss
            diff = diff.tocoo()
            print("Diff from 5pt Poiss sparsity %d nnz   (should be 0) "%diff.nnz )
            #print(" at nonzero locations" + str( hstack((diff.row.reshape(-1,1), diff.col.reshape(-1,1))) ) )

        return A.tocsr(), vertices, docstring


        
