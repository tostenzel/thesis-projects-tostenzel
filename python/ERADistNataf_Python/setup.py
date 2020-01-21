from setuptools import setup

setup(name='ERADistNataf',
      version='2018-01',
      url='https://www.era.bgu.tum.de/en/software/eradist/',
      author='Engineering Risk Analysis Group',
      py_modules=['ERADist', 'ERANataf'],
      install_requires=['numpy', 'scipy'],
      zip_safe=False)
