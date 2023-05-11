from warnings import warn
from enum import Enum
from .mfem_adv_diff import mfem_adv_diff
from .aniso_diff import aniso_diff
from .poiss2D import poiss2D
from .poiss3D import poiss3D
from .adv2D import adv2D
from .coeffjump2D import coeffjump2D
from .coeffjumpSaw2D import coeffjumpSaw2D

'''    
Interface to get matrices from matrix testbed

The matrix refinement level is controlled by the refinement variable.  Thus,
each matrix test bed can form a sequence for a scaling study (refinement =
0,1,2,...), or be a single example from an interesting application
(refinement=0). 

Data types
----------
examples : enumerated class of supported testbed examples 


Methods
-------

get_mat(example, refinement, kwargs) : returns a matrix for the given example
    type and given refinement level, using any testbed specific kwargs


Example
-------
>>> from pyamg import testbed as tb 
>>> kwargs = {'sigma' : 1000.0}
>>> data = tb.get_mat(tb.examples.coeffjumpSaw2D, 0, **kwargs)
>>> data['A']

Add a new testbed interface
---------------------------
 1) Define new enumerated option for examples
 2) Add call in get_mat(...) to your matrix test bed

    If your new matrix test bed requires specific kwargs, it is recommended to
    check for the correctness of those kwargs, to help the user.

'''


# Define enumerated class, listing all the supported matrix examples
class examples(Enum):
    mfem_adv_diff = 1
    firedrake_adv_diff= 2
    aniso_diff = 3
    poiss2D = 4
    poiss3D = 5
    adv2D = 6
    coeffjump2D = 7
    coeffjumpSaw2D = 8


def get_mat(example, refinement, **kwargs):
    '''
    Returns a matrix for the given example type and given refinement level,
    using any testbed specific kwargs

    Parameters
    ----------
    example : enumerated type, testbed.examples
        Set to one of the supported example options in the enumerated type
        testbed.examples.

    refinement : int
        All examples support a refinement of 0. For scaling study examples,
        higher levels of refinement (i.e., larger problems) are supported, and
        refinment can be 0, 1, 2, ... 

    kwargs : dictionary (optional)
        Many testbed examples take (or require) additional parameters, which
        are specified in kwargs.  Each test bed should print a help message if
        called with incorrect kwargs, describing what is required.


    Returns
    -------
    
    Dictionary containing { 'A' : A, 'b' : b }, where A is the matrix and b
    is the right hand side.

    Optional dictionary members are as follows.
        'B' : near null space mode(s)
        'vertices' : spatial points for each degree of freedom (e.g., Lagrangian discretizations)
        'docstring' : description of problem


    Example Testbeds Available
    --------------------------
    >>> from pyamg import testbed as tb 
    >>> for ee in tb.examples: print(ee)


    Example Testbed Usage
    ---------------------
    >>> from pyamg import testbed as tb 
    >>> kwargs = {'sigma' : 1000.0}
    >>> data = tb.get_mat(tb.examples.coeffjumpSaw2D, 0, **kwargs)


    Notes
    -----

    '''
    
    ##
    # Based on the given example type, call the appropriate matrix testbed
    if example == examples.mfem_adv_diff:
        try:
            import mfem.ser as mfem
        except:
            raise NameError("Install PyMFEM for this example")

        data = mfem_adv_diff(refinement, **kwargs)
        return data

    elif example == examples.aniso_diff:
        data = aniso_diff(refinement, **kwargs)
        return data

    elif example == examples.poiss2D:
        data = poiss2D(refinement, **kwargs)
        return data

    elif example == examples.poiss3D:
        data = poiss3D(refinement, **kwargs)
        return data

    elif example == examples.adv2D:
        data = adv2D(refinement, **kwargs)
        return data

    elif example == examples.coeffjump2D:
        data = coeffjump2D(refinement, **kwargs)
        return data

    elif example == examples.coeffjumpSaw2D:
        data = coeffjumpSaw2D(refinement, **kwargs)
        return data

    elif example == examples.firedrake_adv_diff:
        print("Ben of the firedrake examples, implement me!")

    else:
        raise ValueError(f"Unsupported matrix testbed example {example}. Please \
                 use an example present in enumerated type testbed.examples") 


