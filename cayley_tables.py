import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional, Tuple, Union
from itertools import product
import colorsys

class GroupTable:
    """Create interactive, colored Cayley tables for group visualization."""

    def __init__(self, elements: List[str], operation_table: np.ndarray, name: str = "Group"):
        """
        Initialize a group table.

        Args:
            elements: List of group element names (e.g., ['e', 'a', 'b', 'ab'])
            operation_table: 2D numpy array where table[i,j] = index of elements[i] * elements[j]
            name: Name of the group
        """
        self.elements = elements
        self.n = len(elements)
        self.table = operation_table
        self.name = name
        self.identity_idx = self._find_identity()

    def _find_identity(self) -> int:
        """Find the identity element index."""
        for i in range(self.n):
            if all(self.table[i, j] == j and self.table[j, i] == j for j in range(self.n)):
                return i
        return 0  # Default to first element if not found

    def _generate_colors(self, scheme: str = 'rainbow') -> np.ndarray:
        """Generate colors for elements based on different schemes."""
        colors = np.zeros((self.n, self.n))

        if scheme == 'rainbow':
            # Each element gets a distinct hue
            for i in range(self.n):
                for j in range(self.n):
                    colors[i, j] = self.table[i, j]

        elif scheme == 'order':
            # Color by element order
            orders = self._compute_orders()
            for i in range(self.n):
                for j in range(self.n):
                    colors[i, j] = orders[self.table[i, j]]

        elif scheme == 'conjugacy':
            # Color by conjugacy classes
            conj_classes = self._compute_conjugacy_classes()
            for i in range(self.n):
                for j in range(self.n):
                    colors[i, j] = conj_classes[self.table[i, j]]

        elif scheme == 'inverse':
            # Highlight inverse pairs
            for i in range(self.n):
                for j in range(self.n):
                    result = self.table[i, j]
                    # Check if elements are inverses
                    if self.table[i, j] == self.identity_idx:
                        colors[i, j] = 0  # Special color for inverse pairs
                    else:
                        colors[i, j] = result + 1

        return colors

    def _compute_orders(self) -> List[int]:
        """Compute the order of each element."""
        orders = []
        for i in range(self.n):
            power = i
            order = 1
            while power != self.identity_idx and order <= self.n:
                power = self.table[power, i]
                order += 1
            orders.append(order if power == self.identity_idx else self.n)
        return orders

    def _compute_conjugacy_classes(self) -> List[int]:
        """Compute conjugacy classes."""
        classes = [-1] * self.n
        class_id = 0

        for i in range(self.n):
            if classes[i] == -1:
                # Find all conjugates of element i
                conjugates = set()
                for g in range(self.n):
                    # Compute g * i * g^(-1)
                    g_inv = self._find_inverse(g)
                    conjugate = self.table[self.table[g, i], g_inv]
                    conjugates.add(conjugate)

                # Assign same class ID to all conjugates
                for conj in conjugates:
                    classes[conj] = class_id
                class_id += 1

        return classes

    def _find_inverse(self, element_idx: int) -> int:
        """Find the inverse of an element."""
        for i in range(self.n):
            if self.table[element_idx, i] == self.identity_idx:
                return i
        return element_idx  # Return self if no inverse found

    def _get_accessible_colorscale(self, accessibility: str, base_scheme: str) -> Union[str, List]:
        """Get colorscale based on accessibility mode."""

        # Colorblind-friendly palettes
        palettes = {
            'standard': {
                'rainbow': 'Rainbow',
                'other': 'Viridis'
            },
            'deuteranopia': {  # Red-green colorblind (most common)
                'rainbow': [[0, '#0B66FF'], [0.33, '#C2007A'], [0.67, '#FF8C00'], [1, '#7AC943']],
                'other': [[0, '#222222'], [0.25, '#0B66FF'], [0.5, '#C2007A'], [0.75, '#FF8C00'], [1, '#F5F5F7']]
            },
            'protanopia': {  # Red-green colorblind
                'rainbow': [[0, '#0B66FF'], [0.33, '#C2007A'], [0.67, '#FF8C00'], [1, '#7AC943']],
                'other': [[0, '#222222'], [0.25, '#0B66FF'], [0.5, '#C2007A'], [0.75, '#FF8C00'], [1, '#F5F5F7']]
            },
            'tritanopia': {  # Blue-yellow colorblind (rare)
                'rainbow': [[0, '#7AC943'], [0.33, '#C2007A'], [0.67, '#FF8C00'], [1, '#222222']],
                'other': [[0, '#222222'], [0.25, '#7AC943'], [0.5, '#C2007A'], [0.75, '#FF8C00'], [1, '#F5F5F7']]
            },
            'universal': {  # High contrast for all types
                'rainbow': [[0, '#0B66FF'], [0.33, '#FF8C00'], [0.67, '#7AC943'], [1, '#C2007A']],
                'other': [[0, '#222222'], [0.25, '#0B66FF'], [0.5, '#FF8C00'], [0.75, '#7AC943'], [1, '#F5F5F7']]
            }
        }

        mode = palettes.get(accessibility, palettes['standard'])
        if base_scheme == 'rainbow':
            return mode.get('rainbow', 'Rainbow')
        else:
            return mode.get('other', 'Viridis')

    def plot(self, color_scheme: str = 'rainbow',
             show_grid: bool = True,
             highlight_subgroup: Optional[List[int]] = None,
             title: Optional[str] = None,
             show_labels: bool = True,
             color_accessibility: str = 'standard') -> go.Figure:
        """
        Create an interactive Plotly figure of the Cayley table.

        Args:
            color_scheme: 'rainbow', 'order', 'conjugacy', or 'inverse'
            show_grid: Whether to show grid lines
            highlight_subgroup: List of indices to highlight as a subgroup
            title: Custom title for the plot
            show_labels: Whether to show element labels in cells
            color_accessibility: 'standard', 'deuteranopia', 'protanopia', 'tritanopia', or 'universal'

        Returns:
            Plotly figure object
        """
        # Generate color matrix
        colors = self._generate_colors(color_scheme)

        # Select colorscale based on accessibility mode
        colorscale = self._get_accessible_colorscale(color_accessibility, color_scheme)

        # Create hover text
        hover_text = []
        for i in range(self.n):
            row = []
            for j in range(self.n):
                result_idx = self.table[i, j]
                text = f"{self.elements[i]} × {self.elements[j]} = {self.elements[result_idx]}"

                # Add order information
                if color_scheme == 'order':
                    orders = self._compute_orders()
                    text += f"<br>Order of result: {orders[result_idx]}"

                row.append(text)
            hover_text.append(row)

        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=colors,
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title=color_scheme.capitalize(),
                tickmode='array',
                tickvals=list(range(self.n)),
                ticktext=self.elements if color_scheme == 'rainbow' else None
            )
        ))

        # Add text annotations for each cell (only if show_labels is True)
        annotations = []
        if show_labels:
            for i in range(self.n):
                for j in range(self.n):
                    result_idx = self.table[i, j]
                    annotations.append(
                        dict(
                            x=j, y=i,
                            text=self.elements[result_idx],
                            showarrow=False,
                            font=dict(color='white' if colors[i, j] > colors.max()/2 else 'black')
                        )
                    )

        # Highlight subgroup if provided
        if highlight_subgroup:
            for i in highlight_subgroup:
                for j in highlight_subgroup:
                    annotations.append(
                        dict(
                            x=j, y=i,
                            text='',
                            showarrow=False,
                            bordercolor='red',
                            borderwidth=2,
                            bgcolor='rgba(255, 0, 0, 0.1)'
                        )
                    )

        # Update layout
        fig.update_layout(
            title=title or f"{self.name} Cayley Table ({color_scheme} coloring)",
            xaxis=dict(
                title="Second Element",
                tickmode='array',
                tickvals=list(range(self.n)),
                ticktext=self.elements,
                side='bottom',
                showgrid=show_grid
            ),
            yaxis=dict(
                title="First Element",
                tickmode='array',
                tickvals=list(range(self.n)),
                ticktext=self.elements,
                autorange='reversed',
                showgrid=show_grid
            ),
            annotations=annotations,
            width=600,
            height=600
        )

        return fig

    def find_subgroups(self) -> List[List[int]]:
        """Find all subgroups of the group."""
        subgroups = [[self.identity_idx]]  # Identity always forms a subgroup

        # Check all possible subsets
        from itertools import combinations
        for r in range(2, self.n + 1):
            for subset in combinations(range(self.n), r):
                if self.identity_idx not in subset:
                    continue

                # Check if subset is closed under operation
                is_closed = True
                for i in subset:
                    for j in subset:
                        if self.table[i, j] not in subset:
                            is_closed = False
                            break
                    if not is_closed:
                        break

                if is_closed:
                    subgroups.append(list(subset))

        return subgroups

    def analyze(self) -> Dict:
        """Analyze group properties."""
        analysis = {
            'order': self.n,
            'identity': self.elements[self.identity_idx],
            'is_abelian': np.array_equal(self.table, self.table.T),
            'element_orders': {},
            'subgroups': [],
            'conjugacy_classes': {}
        }

        # Element orders
        orders = self._compute_orders()
        for i, order in enumerate(orders):
            analysis['element_orders'][self.elements[i]] = order

        # Subgroups
        subgroups = self.find_subgroups()
        for sg in subgroups:
            analysis['subgroups'].append([self.elements[i] for i in sg])

        # Conjugacy classes
        conj_classes = self._compute_conjugacy_classes()
        class_dict = {}
        for i, class_id in enumerate(conj_classes):
            if class_id not in class_dict:
                class_dict[class_id] = []
            class_dict[class_id].append(self.elements[i])
        analysis['conjugacy_classes'] = list(class_dict.values())

        return analysis


# Example groups
class ExampleGroups:
    """Collection of example groups for visualization."""

    @staticmethod
    def cyclic(n: int) -> GroupTable:
        """Create cyclic group C_n."""
        elements = ["e"] + [str(i) for i in range(1, n)]
        table = np.array([[(i + j) % n for j in range(n)] for i in range(n)])
        return GroupTable(elements, table, f"C_{n}")

    @staticmethod
    def dihedral(n: int) -> GroupTable:
        """Create dihedral group D_n."""
        # Elements: n rotations and n reflections
        elements = ["e"] + [str(i) for i in range(1, n)]
        elements += [f"s{i}" for i in range(n)]

        size = 2 * n
        table = np.zeros((size, size), dtype=int)

        # Rotation × Rotation
        for i in range(n):
            for j in range(n):
                table[i, j] = (i + j) % n

        # Rotation × Reflection
        for i in range(n):
            for j in range(n):
                table[i, n + j] = n + (j - i) % n

        # Reflection × Rotation
        for i in range(n):
            for j in range(n):
                table[n + i, j] = n + (i + j) % n

        # Reflection × Reflection
        for i in range(n):
            for j in range(n):
                table[n + i, n + j] = (j - i) % n

        return GroupTable(elements, table.astype(int), f"D_{n}")

    @staticmethod
    def klein_four() -> GroupTable:
        """Create Klein four-group."""
        elements = ['e', 'a', 'b', 'ab']
        table = np.array([
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0]
        ])
        return GroupTable(elements, table, "Klein Four-Group")

    @staticmethod
    def quaternion() -> GroupTable:
        """Create quaternion group Q_8."""
        elements = ['1', '-1', 'i', '-i', 'j', '-j', 'k', '-k']
        table = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7],  # 1
            [1, 0, 3, 2, 5, 4, 7, 6],  # -1
            [2, 3, 1, 0, 6, 7, 5, 4],  # i
            [3, 2, 0, 1, 7, 6, 4, 5],  # -i
            [4, 5, 7, 6, 1, 0, 2, 3],  # j
            [5, 4, 6, 7, 0, 1, 3, 2],  # -j
            [6, 7, 4, 5, 3, 2, 1, 0],  # k
            [7, 6, 5, 4, 2, 3, 0, 1]   # -k
        ])
        return GroupTable(elements, table, "Quaternion Group Q_8")


# Visualization helper functions
def compare_groups(*groups: GroupTable, color_scheme: str = 'rainbow') -> go.Figure:
    """Create a subplot comparing multiple groups."""
    from plotly.subplots import make_subplots

    n_groups = len(groups)
    fig = make_subplots(
        rows=1, cols=n_groups,
        subplot_titles=[g.name for g in groups],
        horizontal_spacing=0.1
    )

    for idx, group in enumerate(groups, 1):
        colors = group._generate_colors(color_scheme)

        heatmap = go.Heatmap(
            z=colors,
            colorscale='Viridis' if color_scheme != 'rainbow' else 'Rainbow',
            showscale=(idx == n_groups),
            text=[[group.elements[group.table[i, j]] for j in range(group.n)]
                  for i in range(group.n)],
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='%{text}<extra></extra>'
        )

        fig.add_trace(heatmap, row=1, col=idx)

        fig.update_xaxes(
            tickmode='array',
            tickvals=list(range(group.n)),
            ticktext=group.elements,
            row=1, col=idx
        )
        fig.update_yaxes(
            tickmode='array',
            tickvals=list(range(group.n)),
            ticktext=group.elements,
            autorange='reversed',
            row=1, col=idx
        )

    fig.update_layout(
        title_text=f"Group Comparison ({color_scheme} coloring)",
        height=400,
        width=400 * n_groups
    )

    return fig


def interactive_explorer(group: GroupTable) -> None:
    """Create an interactive dashboard for exploring a group."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create subplots for different color schemes
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Rainbow (Element)', 'Order', 'Conjugacy Classes', 'Inverse Pairs'],
        horizontal_spacing=0.12,
        vertical_spacing=0.12
    )

    schemes = ['rainbow', 'order', 'conjugacy', 'inverse']
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for scheme, (row, col) in zip(schemes, positions):
        colors = group._generate_colors(scheme)

        heatmap = go.Heatmap(
            z=colors,
            colorscale='Viridis' if scheme != 'rainbow' else 'Rainbow',
            showscale=False,
            text=[[group.elements[group.table[i, j]] for j in range(group.n)]
                  for i in range(group.n)],
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate=f'{scheme}: %{{text}}<extra></extra>'
        )

        fig.add_trace(heatmap, row=row, col=col)

        fig.update_xaxes(
            tickmode='array',
            tickvals=list(range(group.n)),
            ticktext=group.elements,
            row=row, col=col
        )
        fig.update_yaxes(
            tickmode='array',
            tickvals=list(range(group.n)),
            ticktext=group.elements,
            autorange='reversed',
            row=row, col=col
        )

    fig.update_layout(
        title_text=f"{group.name} - Multiple Perspectives",
        height=800,
        width=800
    )

    fig.show()

    # Print analysis
    analysis = group.analyze()
    print(f"\n{group.name} Analysis:")
    print(f"Order: {analysis['order']}")
    print(f"Identity: {analysis['identity']}")
    print(f"Abelian: {analysis['is_abelian']}")
    print(f"\nElement Orders:")
    for elem, order in analysis['element_orders'].items():
        print(f"  {elem}: {order}")
    print(f"\nSubgroups ({len(analysis['subgroups'])}):")
    for sg in analysis['subgroups']:
        print(f"  {{{', '.join(sg)}}}")
    print(f"\nConjugacy Classes:")
    for cc in analysis['conjugacy_classes']:
        print(f"  {{{', '.join(cc)}}}")