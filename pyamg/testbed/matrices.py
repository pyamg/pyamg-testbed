from warnings import warn
from enum import Enum
from .mfem_adv_diff import mfem_adv_diff
from .aniso_diff import aniso_diff

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
        are specified in kwargs


    Returns
    -------
    
    Dictionary containing { 'A' : A, 'b' : b }, where A is the matrix and b
    is the right hand side.

    Optional dictionary members are as follows.
        'B' : near null space mode(s)
        'vertices' : spatial points for each degree of freedom (e.g., Lagrangian discretizations)
        'docstring' : description of problem

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

    elif example == examples.firedrake_adv_diff:
        print("Ben of the firedrake examples, implement me!")

    else:
        raise ValueError(f"Unsupported matrix testbed example {example}. Please \
                 use an example present in enumerated type testbed.examples") 


