import numpy as np
import scipy as sp
from scipy.io import loadmat
from numpy.linalg import norm
from pyamg import gallery
from pyamg.util.utils import print_table
from generator_pymfem_adv_diff import generator_pymfem_adv_diff

# Requires CF_rootnode branch
from pyamg import energymin_cf_solver, air_solver


##
# Solver params
nlevels = 25
krylov='gmres'
coarse_solver='splu'

##
# Choose number of problem refinements to test
nrefinements_to_test = 3

##
# Choose problem options
option = 'poisson2d'
if option == 'mfem_adv_diff':
    # Use MFEM advection-diffusion 
    problem_type = "div_free_recirc"      # b(x,y)= (x(1-x)(2y-1),-(2x-1)(1-y)y)
    #problem_type = "clockwise_rotation"
    #problem_type = "constant_adv" 
    #problem_type = "diffusion_only"
    
    gamma = 1.0     # diffusion coefficient 
    order = 1       # FEM order
    meshfile = "./inline-quad.mesh"
    generator_kwargs={'gamma':gamma, 'meshfile':meshfile, 
                      'order':order, 'problem_type':problem_type}
    generator = generator_pymfem_adv_diff

elif option == 'poisson2d':
    def generator_poisson2d(prob_refinement, **kwargs):
        n = 2**(5 + prob_refinement)
        A = gallery.poisson((n,n), format='csr')
        b = np.zeros_like((A.shape[0],))
        B = np.ones((A.shape[0],1))
        return {'A':A, 'b':b, 'B':B, 'vertices':None, 'docstring':"2D FD Poisson operator"}

    generator = generator_poisson2d
    generator_kwargs = {}

##
# Start tests
table = [ ["n ", "iter", "piter", 'op comp'] ]
np.random.seed(0)

# Run tests with classic AIR
for i in range(nrefinements_to_test):
    data = generator(i, **generator_kwargs) 
    A = data['A']
    x0 = np.random.rand(A.shape[0],)
    b = np.zeros_like(x0)

    ml = air_solver(A, max_levels=nlevels, 
                    strength = ('classical', {'theta' : 0.25}),
                    restrict=('air',{'degree':2, 'theta':0.05}),
                    interpolation='one_point',  # 'one_point' or 'standard'
                    CF=('RS', {'second_pass': True}),
                    presmoother=('jacobi',{'iterations':2}),
                    postsmoother=('fc_jacobi', {'omega': 1.0, 'iterations': 1, 'withrho': False, 'f_iterations': 2, 'c_iterations': 1}),
                    coarse_solver=coarse_solver,
                    filter_operator = None) #(True, 1e-4)) 
    
    res = []
    x = ml.solve(b, x0=x0, residuals=res, maxiter=100, tol=1e-9) 
    print("\n---- n = %d ----"%A.shape[0])
    print("stand-alone")
    print("  max lvls: %d"%nlevels)
    print("  Iter:     %d"%len(res)) 
    print("  ||r||:    %1.2e"%np.sqrt(np.dot(b-A*x, b-A*x )))
    print("  Op Comp:  %1.2f\n"%ml.operator_complexity()) 
    
    pres = []
    x = ml.solve(b, x0=x0, residuals=pres, maxiter=100, tol=1e-9, accel=krylov)
    print("precon")
    print("  max lvls: %d"%nlevels)
    print("  Iter:     %d"%len(pres)) 
    print("  ||r||:    %1.2e"%np.sqrt(np.dot(b-A*x, b-A*x )))

    table.append([ str(A.shape[0]), str(len(res)), str(len(pres)), "%1.2f"%ml.operator_complexity() ])
table_string = print_table(table, title='  ', delim='&', centering='center', 
                           col_padding=2, header=True, headerchar='-')
print(table_string)
print("\n")


# Run tests with my hacky experimental AIR stuff
table = [ ["n ", "iter", "piter", 'op comp'] ]
np.random.seed(0)

for i in range(nrefinements_to_test):
    data = generator(i, **generator_kwargs) 
    A = data['A']
    x0 = np.random.rand(A.shape[0],)
    b = np.zeros_like(x0)
    B = np.ones((A.shape[0], 1))

    ml2 = energymin_cf_solver(A, B=B, BH=B, max_levels=nlevels,
                              strength = ('classical', {'theta' : 0.25}),
                              symmetry='nonsymmetric',
                              restrict=('AIRplus', {'maxiter':1, 'degree':1}),
                              interpolation='one_point',  # 'one_point' or 'standard'
                              aggregate=('RS', {'second_pass': True}),  #'standard' 'PMIS'  ('RS', {'second_pass': True}),
                              presmoother=('jacobi',{'iterations':2}), 
                              postsmoother=('fc_jacobi', {'omega': 1.0, 'iterations': 1, 'withrho': False, 'f_iterations': 2, 'c_iterations': 1}), 
                              improve_candidates=None,#[ ('jacobi',{'iterations':5}) ],  # Jacobi or None
                              diagonal_dominance=False,
                              keep=True,
                              coarse_solver=coarse_solver)

    res = []
    x = ml2.solve(b, x0=x0, residuals=res, maxiter=100, tol=1e-9) 
    print("---- n = %d ----"%A.shape[0])
    print("stand-alone")
    print("  max lvls: %d"%nlevels)
    print("  Iter:     %d"%len(res)) 
    print("  ||r||:    %1.2e"%np.sqrt(np.dot(b-A*x, b-A*x )))
    print("  Op Comp:  %1.2f\n"%ml2.operator_complexity()) 
    
    pres = []
    x = ml2.solve(b, x0=x0, residuals=pres, maxiter=100, tol=1e-9, accel=krylov)
    print("precon")
    print("  max lvls: %d"%nlevels)
    print("  Iter:     %d"%len(pres)) 
    print("  ||r||:    %1.2e"%np.sqrt(np.dot(b-A*x, b-A*x )))

    table.append([ str(A.shape[0]), str(len(res)), str(len(pres)), "%1.2f"%ml2.operator_complexity() ])

table_string = print_table(table, title='  ', delim='&', centering='center', 
                           col_padding=2, header=True, headerchar='-')
print(table_string)
    


