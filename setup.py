from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.20'
DESCRIPTION = 'Contains useful functions and classes'

# Setting up
setup(
    name="utilsbox",
    version=VERSION,
    author="Vikas Sanwal",
    author_email="<vikassnwl@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['scipy==1.8.0', 
                      'opencv-contrib-python==4.7.0.72', 
                      'boto3==1.26.3', 
                      'requests==2.31.0', 
                      'beautifulsoup4==4.12.2', 
                      'google-cloud-storage==2.9.0',
                      'azure-storage-blob==12.17.0'],
    keywords=['python'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
