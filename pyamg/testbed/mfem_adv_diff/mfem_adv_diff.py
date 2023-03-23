import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix

import mfem.ser as mfem
from mfem.ser import intArray

class mfem_adv_diff:
    '''    
    MFEM testbed matrix interface
    
    Attributes
    ----------
    

    Methods
    -------
    
    __new__(prob_refinement, kwargs) : This function is called upon object 
        creation and returns a data dictionary with the matrix and other 
        items (see function documentation for __new__)

    ComputeMeshSize(self, mesh) : This function computes the maximum and
        minimum mesh sizes for a mesh
    
     mfem_discretization(self, problem, gamma, meshfile, ref_levels, order) : 
        This function uses MFEM to define a simple finite element
        discretization of advection diffusion (stiffness matrix and mass
        matrix) with simple uniform mesh refinement.


    '''

    
    def __new__(self, prob_refinement, **kwargs):
        '''
        Return a PyMFEM discretization

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
        A, g, vertices, M, h_min, h_max = self.mfem_discretization(self, mfem_problem, gamma, meshfile, ref_levels, order)
        b = sp.rand(A.shape[0],1)
        B = np.ones_like(b)
        
        ##
        # Filter small entries in A
        max_A = np.max(np.abs(A.data))
        A.data[ np.abs(A.data)/max_A < 1e-14 ] = 0.0
        A.eliminate_zeros()
        
        
        docstring = f"DG advection diffusion using mesh file {meshfile}, FEM order {order}, diffusion constant {gamma}, and problem type {problem_discr} with " + docstring
        
        return {'A':A, 'b':b, 'B':B, 'vertices':vertices, 'docstring':docstring}
        

    def ComputeMeshSize(self, mesh):
        
        '''
        Description
        ------------
        This function computes the maximum and minimum mesh sizes for a mesh
    
        Input
        ------
        mesh:    Finite element discretization mesh
    
        Output
        ------
        h_min:   Minimum mesh size
    
        h_max:   Maximum mesh size
        
        '''
        
        h_max = -1.0
        h_min = -1.0
        NumOfElements = mesh.GetNE()
    
        for i in range(NumOfElements):
            h = mesh.GetElementSize(i)
            if (i == 0):
                h_min = h_max = h
            else:
                if (h < h_min):  h_min = h
                if (h > h_max):  h_max = h
    
        return (h_min, h_max)

    
    def mfem_discretization(self, problem, gamma, meshfile, ref_levels, order):
    
        '''
        Description
        ------------
        This function uses MFEM to define a simple finite element
        discretization of advection diffusion (stiffness matrix and mass
        matrix) with simple uniform mesh refinement.
    
        Input
        -----
        problem        : parameter to define problem type for various type orientation in advection
        gamma          : User supplied diffusion coefficient
        meshfile       : Identifier for finite element mesh file (.mesh format)
        ref_levels     : Mesh refinement level 
        order          : order of the finite element space
      
        
        Output
        ------
        NumOfElements  : Number of elements in finite element discretization mesh
        K              : Finite element stiffness matrix (explicit)
        M              : Finite element mass matrix (implicit)
        g              : Initial condition
    
        Parameter choices for an example run of this function
        ----------------------------------------------------
    
        problem = 0
        ref_levels = 1
        order = 1
        '''
        
        
        # 1. Read the mesh from the given mesh file. We can handle geometrically
        #    periodic meshes in this code.
        mesh = mfem.Mesh(meshfile, 1,1)
        dim = mesh.Dimension()
        sdim = mesh.SpaceDimension()
    
        
        # 2. Refine the mesh to increase the resolution. In this example we do
        #    'ref_levels' of uniform refinement, where 'ref_levels' is a
        #    input parameter. If the mesh is of NURBS type, we convert it
        #    to a (piecewise-polynomial) high-order mesh.
        for lev in range(ref_levels):
            mesh.UniformRefinement();
            if mesh.NURBSext:
                mesh.SetCurvature(max(order, 1))
                
        bb_min, bb_max = mesh.GetBoundingBox(max(order, 1));
        (h_min, h_max) = self.ComputeMeshSize(self, mesh)
    
     
        # 3. Begin constructing stiffness matrix (K) for advection 
        #    Define coefficient using VectorPyCoefficient and PyCoefficient
        class velocity_coeff(mfem.VectorPyCoefficient):
           def EvalValue(self, x):        
               dim = len(x)
                
               center = (bb_min + bb_max)/2.0
               # map to the reference [-1,1] domain                
               X = 2 * (x - center) / (bb_max - bb_min)
               if problem == 0:
                   # Constant advection 
                   if dim == 1: v = [1.0,]
                   elif dim == 2: v = [np.sqrt(2./3.), np.sqrt(1./3)]
                   elif dim == 3: v = [np.sqrt(3./6.), np.sqrt(2./6), np.sqrt(1./6.)]
               elif problem == 1:
                   # Clockwise rotation in 2D around the origin                
                   w = pi/2
                   if dim == 1: v = [1.0,]
                   elif dim == 2: v = [w*X[1],  - w*X[0]]
                   elif dim == 3: v = [w*X[1],  - w*X[0],  0]
               elif (problem == 2):
                   # Div-Free recirculation
                   if dim == 1: v = [1.0,]
                   elif dim == 2: v=[ X[0]*(1-X[0])*(2*X[1]-1), -(2*X[0]-1)*(1-X[1])*X[1] ]
                   elif dim == 3:
                       print("div-free recirc does not support 3D, using constant advection instead")
                       v = [np.sqrt(3./6.), np.sqrt(2./6), np.sqrt(1./6.)]
               elif (problem == 3):
                   # Pure diffusion
                   if dim == 1: v = [0.0,]
                   elif dim == 2: v = [0.0, 0.0]
                   elif dim == 3: v = [0.0, 0.0, 0.0]
                   
               return v
                
        class rhs_coeff(mfem.PyCoefficient):
           def EvalValue(self, x):        
               dim = len(x)
               center = (bb_min + bb_max)/2.0
               # map to the reference [-1,1] domain        
               X = 2 * (x - center) / (bb_max - bb_min)
               return np.sin(np.pi * X[0]) * np.sin(np.pi * X[1])
    
    
        # 4. Inflow boundary condition (zero for the problems considered in this code)
        class inflow_coeff(mfem.PyCoefficient):
           def EvalValue(self, x):
               return 0
    
    
        # 5. Define the discontinuous DG finite element space of the given
        #    polynomial order on the refined mesh.
        fec = mfem.DG_FECollection(order, dim)
        fes = mfem.FiniteElementSpace(mesh, fec)
    
    
        # 6. Set up and assemble the bilinear and linear forms corresponding to the
        #    DG discretization for Advection. The DGTraceIntegrator involves integrals
        #    over mesh interior faces.
        velocity = velocity_coeff(dim)
        inflow = inflow_coeff()
        m = mfem.BilinearForm(fes)
        m.AddDomainIntegrator(mfem.MassIntegrator())
        k = mfem.BilinearForm(fes)
        k.AddDomainIntegrator(mfem.ConvectionIntegrator(velocity, -1.0))
        k.AddInteriorFaceIntegrator(
              mfem.TransposeIntegrator(mfem.DGTraceIntegrator(velocity, 1.0, -0.5)))
        k.AddBdrFaceIntegrator(
              mfem.TransposeIntegrator(mfem.DGTraceIntegrator(velocity, 1.0, -0.5)))
    
        m.Assemble()
        m.Finalize()
        skip_zeros = 0
        k.Assemble(skip_zeros)
        k.Finalize(skip_zeros)
        
        # Computes RHS inflow conditions (not used here)
        #b = mfem.LinearForm(fes)
        #b.AddBdrFaceIntegrator(
        #      mfem.BoundaryFlowIntegrator(inflow, velocity, -1.0, -0.5)) 
        #b.Assemble()
    
    
        # 7. Generate Scipy Sparse CSR stiffness matrix for advection and RHS vector
        Kmfem = k.SpMat()
        Mmfem = m.SpMat()
        K = csr_matrix( (Kmfem.GetDataArray().copy(), Kmfem.GetJArray().copy(), Kmfem.GetIArray().copy()) )   
        M = csr_matrix( (Mmfem.GetDataArray().copy(), Mmfem.GetJArray().copy(), Mmfem.GetIArray().copy()) )   
        rhs_fcn = rhs_coeff()
        g = mfem.GridFunction(fes)
        g.ProjectCoefficient(rhs_fcn)
        g = g.GetDataArray().copy()
    
    
        # 8. IP method for Diffusion operator
        #
        #   Set up the bilinear form a(.,.) on the finite element space
        #   corresponding to the Laplacian operator -Delta, by adding the Diffusion
        #   domain integrator and the interior and boundary DG face integrators.
        #   Note that boundary conditions are imposed weakly in the form, so there
        #   is no need for dof elimination. After assembly and finalizing we
        #   extract the corresponding sparse matrix Dmfem.
        sigma = -1.0
        kappa = (order+1)**2.
    
        if gamma == 0:
            D = csr_matrix( ([], ([],[])), (K.shape[0],K.shape[1]) )
        else:
    
            # Not currently used
            #   Could try and set a nonzero Dirichlet boundary (to be added to RHS) with dbcCoef
            #   and ProjectCoefficient, like is done above for g.
            #dbc_val=0.0
            #dbcCoef = mfem.ConstantCoefficient(dbc_val) 
            
            # Define weak Dirichlet boundaries over whole Domain for diffusion
            #   Create "marker arrays" to define the portions of boundary associated
            #   with each type of boundary condition. These arrays have an entry
            #   corresponding to each boundary attribute.  Placing a '1' in entry i
            #   marks attribute i+1 as being active, '0' is inactive.
            dbc_bdr = mfem.intArray(mesh.bdr_attributes.Max()) 
            dbc_bdr.Assign(0)
            dbc_bdr[1] = 1
    
            gamma_coeff = mfem.ConstantCoefficient(gamma)
            a = mfem.BilinearForm(fes)
            a.AddDomainIntegrator(mfem.DiffusionIntegrator(gamma_coeff))
            a.AddInteriorFaceIntegrator(mfem.DGDiffusionIntegrator(gamma_coeff, sigma, kappa))
            a.AddBdrFaceIntegrator(mfem.DGDiffusionIntegrator(gamma_coeff, sigma, kappa),dbc_bdr)
            a.Assemble()
            a.Finalize()
            Dmfem = a.SpMat()
            D = csr_matrix( (Dmfem.GetDataArray().copy(), Dmfem.GetJArray().copy(), Dmfem.GetIArray().copy()) )
    
    
        # 9. Eliminate zeros from all the matrices
        M.eliminate_zeros()
        K.eliminate_zeros()
        D.eliminate_zeros()
        
    
        # 10. Compile vector of vertices
        vertices = np.zeros((fes.GetTrueVSize(), sdim))
        R = fes.GetConformingRestriction()
        if R is not None:    
            VDof2TDof = np.zeros(fes.GetNDofs(), dtype=int)
            for i, j in enumerate(R.GetJArray()):
                VDof2TDof[j] = i
            TDof2Vdof = R.GetJArray().copy()
        else:
            VDof2TDof = None
            TDof2VDof = None
            
        for j in range(fes.GetNE()):
            el = fes.GetFE(j)
            tr = fes.GetElementTransformation(j)
            vdofs = fes.GetElementVDofs(j)
            
            tdofs= vdofs if VDof2TDof is None else [VDof2TDof[k] for k in vdofs]
            
            ir = el.GetNodes()
            for k, tdof in enumerate(tdofs):
                vertices[tdof] = tr.Transform(ir.IntPoint(k))
        ##
        NumOfElements = mesh.GetNE()
    
     
        ##
        # This will print the mesh to file
        #mesh.Print('refined.mesh', 8)
        
        return (D+K, g, vertices, M, h_min, h_max)
