from setuptools import setup, find_packages

packages=find_packages(include=['networks', 'networks.*', 'modules', "modules.*"]),

with open('requirements.txt') as f:
    required = f.read().splitlines()

required.pop() # removes the local package
print(required)
    
setup(
   name='shared_modules',
   version='0.1.1',
   description='Shared components for cspca segmentation',
   author='Michael Staff Larsen and Syed Farhan Abbas',
   author_email='michael.s.larsen@ntnu.no',
   packages=['shared_modules'],  
)