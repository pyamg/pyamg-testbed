# pyamg-testbed
Experimental testbed for PyAMG with various matrix generators

# installation
From source directory, 

    pip install .

# usage
Install any dependencies for your testbed (like PyMFEM or FireDrake)

   pip install pymfem

Call testbed for your example to generate matrices

   from pyamg import testbed as tb
   args = {'gamma' : 0.1, 'meshfile' : 'pathtomesh/inline-quad.mesh', 'order' : 1}
   data = tb.get_mat(tb.examples.mfem_adv_diff, 0, **args)
   print(data['A'])
   print(data['b'])
   print(data['docstring'])

# add new testbed matrix example
See `matrices.py`


