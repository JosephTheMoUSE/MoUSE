# If you want to propagate functionality from your python module (example_utils.py) one level higher # noqa
# you should include something like this in init:
#
# from .example_utils import * (or function(s) that you want to propagate)
#
# this will make import look like (suppose we have `example` function in example_utils.py): (1) # noqa
# from mouse.utils import example
#
# instead of: (2)
# from mouse.utils.example_utils import example
#
# Note: adding import in `init` will still allow for (2)
#       but be careful with import * because it may introduce shadowing of functions # noqa
