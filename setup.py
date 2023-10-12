#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    author="Qubit Ulm",
    author_email='qubit-ulm@example.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Clifford simulations for error correction",
    install_requires=requirements,
    license="Apache Software License 2.0",
    include_package_data=True,
    keywords='error_correction_sim',
    name='error_correction_sim',
    packages=find_packages(include=['error_correction_sim', 'error_correction_sim.*']),
    url='https://github.com/qubit-ulm/error_correction_sim',
    version='0.1.0',
    zip_safe=False,
)
