## pyamg-testbed
Experimental testbed for PyAMG with various matrix generators

## installation
From source directory, 

    $ pip install .

## usage
Install any dependencies for your testbed (like PyMFEM or FireDrake)

    $ pip install pymfem

Call testbed for your example to generate matrices

    >>> from pyamg import testbed as tb
    >>> meshpath = tb.__path__[0] + '/mfem_adv_diff/inline-quad.mesh'
    >>> args = {'gamma' : 0.1, 'meshfile' : meshpath, 'order' : 1} 
    >>> data = tb.get_mat(tb.examples.mfem_adv_diff, 0, **args)
    >>> print(data.keys())
    >>> print(data['docstring'])

## add new testbed matrix example
See `matrices.py`


