import numpy

from distutils.core import setup
from Cython.Build import cythonize


setup(
    ext_modules=cythonize("cython/fast_bleu.pyx"),
    include_dirs=[numpy.get_include()]
)

setup(
    ext_modules=cythonize("cython/fast_bleu1.pyx"),
    include_dirs=[numpy.get_include()]
)

setup(
   ext_modules=cythonize("cython/fast_gleu.pyx"),
   include_dirs=[numpy.get_include()]
)

setup(
    ext_modules=cythonize("cython/fast_chunkscore.pyx"),
    include_dirs=[numpy.get_include()]
)

setup(
   ext_modules=cythonize("cython/fast_bleu_ref_rollout.pyx"),
   include_dirs=[numpy.get_include()]
)

setup(
   ext_modules=cythonize("cython/fast_bleu_ref_rollout_with_suffix_length.pyx"),
   include_dirs=[numpy.get_include()]
)

setup(
   ext_modules=cythonize("cython/fast_gleu_ref_rollout_with_suffix_length.pyx"),
   include_dirs=[numpy.get_include()]
)

setup(
   ext_modules=cythonize("cython/fast_bleu_ref_rollout_with_suffix_length_bleu1noBrev.pyx"),
   include_dirs=[numpy.get_include()]
)

setup(
   ext_modules=cythonize("cython/fast_bleu_ref_rollout_with_suffix_length_bleu1.pyx"),
   include_dirs=[numpy.get_include()]
)
