Installation
============

Prerequisites
-------------

- Python 3.10 or newer
- R 4.2.2 or newer
- R packages used by the plotting scripts, including ``tidyverse``, ``ComplexHeatmap``, ``BiocManager``, ``reshape``, ``optparse``, and ``utils``

Install from the repository
---------------------------

.. code-block:: bash

   git clone https://github.com/mcgilldinglab/CellAgentChat.git
   cd CellAgentChat
   pip install .

Optional PyTorch support
------------------------

The neural-network workflow depends on PyTorch. Install it separately for your platform or use the optional extra:

.. code-block:: bash

   pip install .[torch]

Notes
-----

- The project now uses ``pyproject.toml`` for packaging.
- Mesa is intentionally pinned to ``1.0.0`` for API stability.
