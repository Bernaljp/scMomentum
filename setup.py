from setuptools import setup, find_packages

setup(
    name='scMomentum',
    version='0.1.0',
    author='Juan Pablo Bernal-Tamayo',
    author_email='juan.bernaltamayo@kaust.edu.sa',
    packages=find_packages(),
    description='A package for single-cell momentum analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=open('requirements.txt').readlines(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
