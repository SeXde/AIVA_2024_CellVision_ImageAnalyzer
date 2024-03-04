from setuptools import find_packages
from setuptools import setup

setup(
    name='CellVision ImageAnalyzer',
    version='0.1',
    license='GPL-3.0',
    description='Cell detection project.',
    author='Alvaro Berdote Jiménez y Jorge Moreno Fernández.',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,
)
