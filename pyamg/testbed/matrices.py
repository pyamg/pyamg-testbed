from warnings import warn
from enum import Enum
from .mfem_adv_diff import mfem_adv_diff

'''    
Interface to get matrices from matrix testbed using built-in interfaces.

Each matrix test bed can form a sequence for a scaling study, or be a single
example from an interesting application.

Data types
----------
examples : enumerated class of supported testbed examples 


Methods
-------
get_mat(example, refinement, kwargs) : returns a matrix, for a certain level of refinement, and testbed specific kwargs


Add a new testbed interface
---------------------------
 1) Define new enumerated option for examples
 2) Add call in get_mat(...) to your matrix test bed

'''


# Define enumerated class, listing all the supported matrix examples
class examples(Enum):
    mfem_adv_diff = 1
    firedrake_adv_diff= 2


def get_mat(example, refinement, **kwargs):
    '''
    some doc...

    Parameters
    ----------
    example : enumerated type, testbed.examples
        Set to one of the supported examples in the enumerated type
        testbed.examples.

    refinement : int
        All supported examples support a refinement of 0. For scaling
        study examples, higher levels of refinement (i.e., larger problems)
        are supported, and refinment can be 0, 1, 2, ... 

    kwargs : dictionary (optional)
        Many testbed examples take (or require) additional parameters, which
        are specified in kwargs


    Returns
    -------
    
    Dictionary containing { 'A' : A, 'b' : b }, where A is the matrix and b
    is the right hand side.

    Optional dictionary members are as follows.
    'B' : near null space mode(s)
    'vertices' : spatial points for each degree of freedom (for Lagrangian discretizations)
    'docstring' : description of problem

    Notes
    -----

    '''
    
    ##
    # Based on the example, call the appropriate matrix testbed
    if example == examples.mfem_adv_diff:
        # inside the call here, check for pymfem, and check kwargs...
        data = mfem_adv_diff(0, **kwargs)
        return data

    elif example == examples.firedrake_adv_diff:
        print("Ben of the firedrake examples, implement me!")

    else:
        raise ValueError(f"Unsupported matrix testbed example {example}. Please \
                 use an example present in enumerated type testbed.examples") 


