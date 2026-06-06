Installation
==============

Installing L-Py
***************

``L-Py`` distribution is based on the ``conda`` software environment management system.
To install conda, you may refer to its installation page: https://docs.conda.io/projects/conda/en/latest/user-guide/install/


Installing binaries using conda



To install L-Py, you need to create an environment (named lpy in this case) :

.. code-block:: bash

        conda create -n lpy openalea.lpy -c fredboudon -c conda-forge

The package is retrieved from the ``fredboudon`` channel (developement) and its dependencies will be taken from ``conda-forge`` channel.

Then, you need to activate the L-Py environment

.. code-block:: bash

        conda activate lpy

And then run L-Py

.. code-block:: bash

        lpy

For any issues with L-py, please check the documentation of L-Py provided here https://lpy.readthedocs.io/en/latest/user/installing.html


Installing TreeSim_Lpy
***********************

With the conda environment for L-Py set, next we need to clone the TreeSim_Lpy repository. To do that run

.. code-block:: bash

        git clone https://github.com/OSUrobotics/treesim_lpy.git