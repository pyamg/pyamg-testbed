from mfem_discretization import mfem_discretization
import numpy as np
import scipy as sp

def generator_pymfem_adv_diff(prob_refinement, **kwargs):
    '''
    Input
    -----
    prob_refinement : refinement of problem (0, 1, 2, ...) to generate
    
    kwargs : dictionary of MFEM parameters, must contain the following keys
    
      'gamma' : float 
                diffusion constant

      'meshfile' : string
                   filename for mesh
      
      'order' : int
                finite-element order
      
      'problem_type' : string, 
                       default is 'constant_adv', but can be one of
                      'div_free_recirc', 'clockwise_rotation', 'constant_adv', 'diffusion_only'
    
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

    ref_levels = prob_refinement + 2
    
    ##
    # Retrieve discretization paramters
    try:
        gamma = kwargs ['gamma']
        meshfile = kwargs['meshfile']
        order = kwargs['order']
    except:
        raise ValueError("Incorrect kwargs for pymfem generator")    

    ##
    # Retrieve problem type
    try:
        problem_discr = kwargs['problem_type']
    except:
        problem_discr = 'constant_adv'
    #
    print(f"Using problem type {problem_discr}")


    ##
    # Generate spatial discretization matrix
    if problem_discr == "constant_adv":
        mfem_problem = 0
        docstring = "constant advection speed over the domain, b(x,y) = [sqrt(2/3),  sqrt(1/3)]"

    elif problem_discr == "clockwise_rotation":
        mfem_problem = 1
        docstring = "clockwise advection rotation around the origin, b(x,y) = [pi/2 y,  -pi/2 x]"

    elif problem_discr == "div_free_recirc":
        mfem_problem = 2
        docstring = "div-free recirculating advection, b(x,y)= [x(1-x)(2y-1),  -(2x-1)(1-y)y]"

    elif problem_discr == "diffusion_only":
        mfem_problem = 3
        docstring = "no advection (diffusion only)"
    else:
        raise AssertionError(f"Invalid problem description {problem_discr}")

    ##
    # Generate matrices and near nullspace modes
    A, g, vertices, M, h_min, h_max = mfem_discretization(mfem_problem, gamma, meshfile, ref_levels, order)
    b = sp.rand(A.shape[0],1)
    B = np.ones_like(b)
    
    ##
    # Filter small entries in A
    max_A = np.max(np.abs(A.data))
    A.data[ np.abs(A.data)/max_A < 1e-14 ] = 0.0
    A.eliminate_zeros()


    docstring = f"DG advection diffusion using mesh file {meshfile}, FEM order {order}, diffusion constant {gamma}, and problem type {problem_discr} with " + docstring
    
    return {'A':A, 'b':b, 'B':B, 'vertices':vertices, 'docstring':docstring}

