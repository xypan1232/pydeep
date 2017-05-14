from setuptools import setup, find_packages

setup(name='pydeep',
    version='0.1',
    description='Convolutional neural network based prediction of protein-RNA binding sites and motifs.',
    url='http://github.com/pydeep',
    author='Xiaoyong Pan',
    author_email='xypan172436@gmail.com',
    license='MIT',
    packages=['pydeep'],
    install_requires=['pandas',
    'numpy',
    'pytorch',
    'matplotlib'
    ],
    scripts=['bin/pydeep'],
    zip_safe=False)
