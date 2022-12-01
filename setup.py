from setuptools import setup, find_packages

setup(
    name='tf-utils',
    version='0.0.1',
    url='https://github.com/hoyso48/tf-utils.git',
    author='hoyso48',
    author_email='hoyeol0730@gmail.com',
    description='utilties for tensorflow 2.x.x',
    packages=["tf_utils"],    
    install_requires=['tensorflow >= 2.0.0'],
)
