from distutils.core import setup
from Cython.Build import cythonize
 
setup(
  name = 'utils',
  ext_modules=cythonize("utils.pyx")
)

setup(
  name = '_tree',
  ext_modules=cythonize("_tree.pyx")
)
