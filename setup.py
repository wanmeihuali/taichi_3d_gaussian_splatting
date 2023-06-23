from setuptools import setup, find_packages
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
setup(
    name="taichi_3d_gaussian_splatting",
    version='0.0.1',
    author="wanmeihuali",
    author_email="kuangyuansun@gmail.com",
    description="An Unofficial Implementation of 3D Gaussian Splatting using Taichi Language",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wanmeihuali/taichi_3d_gaussian_splatting.git",
    packages=find_packages(),
    # install torch usually breaks the environment, so we install requirements.txt manually
    #install_requires=requirements
)