Input Preparation
=================

Ligand-receptor database
------------------------

Provide a CSV, TSV, or text file with three columns:

- ligand-receptor pair identifier
- ligand gene symbol
- receptor gene symbol

Human and mouse databases are bundled under ``src/``.

Gene expression
---------------

Provide a cell-by-gene matrix containing either counts or already normalized expression values.

Metadata
--------

Legacy workflows expect a metadata table containing:

- cell identifier
- ``cell_type``
- ``Batch``

Spatial coordinates
-------------------

Optional spatial coordinates may be supplied for spatial workflows.

- legacy workflows use ``x`` and ``y``

Pseudotime
----------

Optional pseudotime values may be supplied directly or inferred separately.

AnnData / H5AD
--------------

CellAgentChat can work from an AnnData object containing:

- expression matrix in ``X``
- cell annotations in ``obs``
- spatial coordinates in ``obs``
