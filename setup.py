import setuptools

setuptools.setup(
      name='purestochastic',
      version='0.0',
      packages=setuptools.find_packages(),
      author='Victor Bertret',
      
      author_email='victor.bertret@purecontrol.com',
      description='A tensorflow addon for stochastic models',
      
      long_description=open('README.md').read(),
      
      include_package_data=True,
      
      url='https://github.com/purecontrol/purestochastic',
      
      license='GPLv3',
     )
