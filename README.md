# Cayley Tables - Interactive Group Visualization

Visually intuitive, colored Cayley tables for group theory exploration using Plotly.

## Features

- **Interactive Visualizations**: Hover to see group operations
- **Multiple Color Schemes**:
  - Rainbow: Each element gets a distinct color
  - Order: Color by element order
  - Conjugacy: Color by conjugacy classes
  - Inverse: Highlight inverse pairs
- **Group Analysis**: Automatically compute subgroups, element orders, conjugacy classes
- **Subgroup Highlighting**: Visualize subgroups and cosets
- **Built-in Examples**: Cyclic, Dihedral, Klein Four, Quaternion groups and more

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from cayley_tables import ExampleGroups

# Create a dihedral group D_4 (symmetries of a square)
d4 = ExampleGroups.dihedral(4)

# Visualize with conjugacy class coloring
fig = d4.plot(color_scheme='conjugacy')
fig.show()

# Analyze group properties
analysis = d4.analyze()
print(f"Is abelian: {analysis['is_abelian']}")
print(f"Subgroups: {analysis['subgroups']}")
```

## Run Examples

```bash
python examples.py
```

This will demonstrate:
- Cyclic and Dihedral groups
- Klein Four-Group vs Z₂ × Z₂ isomorphism
- Symmetric group S₃
- Quaternion group Q₈
- Lagrange's theorem visualization
- Coset visualization

## Creating Custom Groups

```python
from cayley_tables import GroupTable
import numpy as np

# Define elements and operation table
elements = ['e', 'a', 'b', 'ab']
table = np.array([
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 1, 0]
])

# Create and visualize
group = GroupTable(elements, table, "My Group")
fig = group.plot(color_scheme='order')
fig.show()
```

## Color Schemes Explained

- **Rainbow**: Best for seeing overall structure and patterns
- **Order**: Reveals cyclic subgroups and element periods
- **Conjugacy**: Shows equivalence classes under conjugation
- **Inverse**: Identifies self-inverse elements and inverse pairs
