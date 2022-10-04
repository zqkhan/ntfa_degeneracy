NTFA
====

A generative framework to visualize and analyze individual variation and multiple solutions in fMRI data.

Installation
------------
We recommend setting up a fresh conda environment.

``conda env create -n env_name``

Once done, install the following remaining package directly from source using pip:
  * `Probtorch`_

.. _Probtorch: https://github.com/probtorch/probtorch

``pip install git+https://github.com/probtorch/probtorch.git``

Next, install NTFA from source using pip:

``pip install git+https://github.com/zqkhan/ntfa_degeneracy.git@degeneracy_v2``