from setuptools import find_packages, setup

setup(
    name='KeySan',
    packages=find_packages(include=['keysan']),
    version='0.1.0',
    description='KeySan is a library to get keyphrases using regular expressions or by getting the most recurrent n-grams',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Alejandro Torres Hernández',
    author_email="alejandro.torres9622@gmail.com",
    license='MIT',
    install_requires=[
        "nltk",
        "inflect",
        "scikit-learn"
    ],
)