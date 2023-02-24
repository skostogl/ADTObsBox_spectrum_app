from setuptools import setup

setup(
    name='ADTObsBox_spectrum_app',
    version='1.0',
    description='An app to plot ADTObsBox spectra from NXCALS',
    author='S. Kostoglou',
    author_email='sofia.kostoglou@cern.ch',
    packages=['ADTObsBox_spectrum_app'],
    package_data={'ADTObsBox_spectrum_app': ['*.ui']},
    install_requires=[
        'nxcals',
        'jupyterlab',
        'matplotlib',
        'pandas',
        'pyarrow',
        'scipy',
        'dask',
        'PyQt5',
        'numpy',
        'acc-py-pip-config'
    ],
    entry_points={
        'console_scripts': [
            'ADTObsBox_spectrum_app = ADTObsBox_spectrum_app.main:main'
        ]
    },
)

