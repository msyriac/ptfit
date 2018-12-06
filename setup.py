from distutils.core import setup, Extension
import os



setup(name='ptfit',
      version='0.1',
      description='Point source fitting',
      url='https://github.com/msyriac/ptfit',
      author='Mathew Madhavacheril',
      author_email='mathewsyriac@gmail.com',
      license='BSD-2-Clause',
      packages=['ptfit'],
      package_dir={'ptfit':'ptfit'},
      zip_safe=False)
