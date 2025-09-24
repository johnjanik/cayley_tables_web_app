#!/usr/bin/env python3
"""Interactive web application for exploring group Cayley tables."""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from cayley_tables import GroupTable, ExampleGroups
import numpy as np

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define available groups and their parameters
GROUPS = {
    'cyclic': {'name': 'Cyclic Group', 'params': ['order'], 'default': 5},
    'dihedral': {'name': 'Dihedral Group', 'params': ['order'], 'default': 4},
    'klein': {'name': 'Klein Four-Group', 'params': [], 'default': None},
    'quaternion': {'name': 'Quaternion Group Q₈', 'params': [], 'default': None},
    's3': {'name': 'Symmetric Group S₃', 'params': [], 'default': None},
    'a4': {'name': 'Alternating Group A₄', 'params': [], 'default': None},
    'z2z2': {'name': 'Z₂ × Z₂', 'params': [], 'default': None},
    'dicyclic': {'name': 'Dicyclic Group', 'params': ['order'], 'default': 3},
    'product': {'name': 'Product Group (G × H)', 'params': ['product'], 'default': None},
    'semidirect': {'name': 'Semi-direct Product (N ⋊ H)', 'params': ['semidirect'], 'default': None},
}

# Define available groups for product construction
PRODUCT_GROUPS = {
    'Z2': 'Z₂ (Cyclic order 2)',
    'Z3': 'Z₃ (Cyclic order 3)',
    'Z4': 'Z₄ (Cyclic order 4)',
    'Z5': 'Z₅ (Cyclic order 5)',
    'Z6': 'Z₆ (Cyclic order 6)',
    'Z7': 'Z₇ (Cyclic order 7)',
    'Z8': 'Z₈ (Cyclic order 8)',
    'Z9': 'Z₉ (Cyclic order 9)',
    'Z10': 'Z₁₀ (Cyclic order 10)',
    'Z11': 'Z₁₁ (Cyclic order 11)',
    'Z12': 'Z₁₂ (Cyclic order 12)',
    'Klein': 'Klein Four-Group',
    'S3': 'S₃ (Symmetric)',
    'A4': 'A₄ (Alternating)',
    'D3': 'D₃ (Dihedral order 3)',
    'D4': 'D₄ (Dihedral order 4)',
    'Q8': 'Q₈ (Quaternion)',
}

# Define common semi-direct product actions
SEMIDIRECT_ACTIONS = {
    'trivial': 'Trivial (reduces to direct product)',
    'inversion': 'Inversion (h acts by n → -n)',
    'conjugation': 'Conjugation (for non-abelian H)',
    'dihedral': 'Dihedral action (creates dihedral groups)',
    'automorphism': 'Non-trivial automorphism',
}

COLOR_SCHEMES = {
    'rainbow': 'Rainbow (Elements)',
    'order': 'Element Order',
    'conjugacy': 'Conjugacy Classes',
    'inverse': 'Inverse Pairs'
}

def create_s3():
    """Create symmetric group S_3."""
    s3_elements = ['e', '(12)', '(13)', '(23)', '(123)', '(132)']
    s3_table = np.array([
        [0, 1, 2, 3, 4, 5],  # e
        [1, 0, 4, 5, 2, 3],  # (12)
        [2, 5, 0, 4, 3, 1],  # (13)
        [3, 4, 5, 0, 1, 2],  # (23)
        [4, 3, 1, 2, 5, 0],  # (123)
        [5, 2, 3, 1, 0, 4]   # (132)
    ])
    return GroupTable(s3_elements, s3_table, "S₃")

def create_a4():
    """Create alternating group A_4 (tetrahedral group)."""
    # A4 has 12 elements: identity, 8 3-cycles, and 3 products of disjoint 2-cycles
    a4_elements = ['e', '123', '132', '124', '142', '134', '143',
                   '234', '243', '1234', '1324', '1423']
    a4_table = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # e
        [1, 2, 0, 6, 8, 7, 3, 10, 5, 11, 4, 9],  # 123
        [2, 0, 1, 7, 5, 8, 4, 9, 6, 10, 11, 3],  # 132
        [3, 4, 6, 0, 1, 11, 8, 5, 2, 7, 9, 10],  # 124
        [4, 6, 3, 1, 2, 9, 11, 0, 10, 5, 8, 7],  # 142
        [5, 8, 7, 10, 11, 0, 1, 2, 3, 4, 6, 9],  # 134
        [6, 3, 4, 2, 9, 10, 0, 11, 1, 8, 7, 5],  # 143
        [7, 5, 8, 9, 10, 1, 2, 3, 0, 11, 9, 4],  # 234
        [8, 7, 5, 11, 6, 2, 10, 4, 9, 3, 0, 1],  # 243
        [9, 10, 11, 5, 7, 4, 9, 8, 11, 0, 1, 2],  # 1234
        [10, 11, 9, 8, 3, 6, 5, 1, 4, 2, 0, 8],  # 1324
        [11, 9, 10, 4, 8, 3, 7, 6, 7, 1, 2, 0]   # 1423
    ])
    return GroupTable(a4_elements, a4_table, "A₄")

def create_s4():
    """Create symmetric group S_4."""
    # S4 has 24 elements - for simplicity, using numbered representation
    s4_elements = [str(i) for i in range(24)]
    # This is a simplified representation - full S4 table would be 24x24
    # Using identity and basic structure for demonstration
    s4_table = np.zeros((24, 24), dtype=int)
    for i in range(24):
        for j in range(24):
            # Simplified multiplication - in practice would need full permutation multiplication
            s4_table[i, j] = (i * j) % 24
    return GroupTable(s4_elements, s4_table, "S₄")

def create_dicyclic(n):
    """Create dicyclic group Dic_n of order 4n."""
    # Dic_n = ⟨a, x | a^(2n) = 1, x^2 = a^n, xax^(-1) = a^(-1)⟩
    order = 4 * n
    elements = []

    # Elements are a^i and xa^i for i = 0, ..., 2n-1
    for i in range(2 * n):
        if i == 0:
            elements.append('e')
        else:
            elements.append(str(i))
    for i in range(2 * n):
        elements.append(f'x{i}')

    table = np.zeros((order, order), dtype=int)

    # Define multiplication rules
    for i in range(order):
        for j in range(order):
            if i < 2*n and j < 2*n:
                # a^i * a^j = a^(i+j mod 2n)
                table[i, j] = (i + j) % (2*n)
            elif i < 2*n and j >= 2*n:
                # a^i * xa^j = xa^(j-i)
                j_val = j - 2*n
                table[i, j] = 2*n + ((j_val - i) % (2*n))
            elif i >= 2*n and j < 2*n:
                # xa^i * a^j = xa^(i-j)
                i_val = i - 2*n
                table[i, j] = 2*n + ((i_val - j) % (2*n))
            else:
                # xa^i * xa^j = a^(n+i+j)
                i_val = i - 2*n
                j_val = j - 2*n
                table[i, j] = (n + i_val + j_val) % (2*n)

    return GroupTable(elements, table, f"Dic_{n}")

def create_z2z2():
    """Create Z_2 × Z_2."""
    z2z2_elements = ['(0,0)', '(0,1)', '(1,0)', '(1,1)']
    z2z2_table = np.array([
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0]
    ])
    return GroupTable(z2z2_elements, z2z2_table, "Z₂ × Z₂")

def create_product_group(group1_type, group2_type):
    """Create the direct product of two groups."""
    # Create individual groups
    group1 = get_group_by_type(group1_type)
    group2 = get_group_by_type(group2_type)

    if group1 is None or group2 is None:
        return None

    # Get elements and tables
    elements1 = group1.elements
    elements2 = group2.elements
    table1 = group1.table
    table2 = group2.table

    n1 = len(elements1)
    n2 = len(elements2)
    n = n1 * n2

    # Limit size to prevent browser crashes
    if n > 100:
        return None  # Too large for visualization

    # Create product elements with simplified notation
    product_elements = []
    for i in range(n1):
        for j in range(n2):
            elem1 = elements1[i]
            elem2 = elements2[j]
            # Simplify notation: (e,e) -> e, (e,1) -> e1, (1,e) -> 1e, (1,2) -> 12
            if elem1 == "e" and elem2 == "e":
                product_elements.append("e")
            elif elem1 == "e":
                product_elements.append(f"e{elem2}")
            elif elem2 == "e":
                product_elements.append(f"{elem1}e")
            else:
                product_elements.append(f"{elem1}{elem2}")

    # Create product table
    product_table = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            # Get indices in original groups
            i1, i2 = divmod(i, n2)
            j1, j2 = divmod(j, n2)

            # Multiply in each component
            result1 = table1[i1, j1]
            result2 = table2[i2, j2]

            # Convert back to product index
            product_table[i, j] = result1 * n2 + result2

    group_name = f"{group1.name} × {group2.name}"
    return GroupTable(product_elements, product_table, group_name)

def create_semidirect_product(normal_type, acting_type, action_type):
    """Create the semi-direct product of two groups N ⋊ H."""
    # Create individual groups
    normal_group = get_group_by_type(normal_type)
    acting_group = get_group_by_type(acting_type)

    if normal_group is None or acting_group is None:
        return None

    # Get elements and tables
    n_elements = normal_group.elements
    h_elements = acting_group.elements
    n_table = normal_group.table
    h_table = acting_group.table

    n_size = len(n_elements)
    h_size = len(h_elements)
    total_size = n_size * h_size

    # Define the action of H on N (automorphism)
    def get_action(h_idx, n_idx):
        """Apply the action of h on n."""
        if action_type == 'trivial':
            # Trivial action: h(n) = n for all h, n
            return n_idx
        elif action_type == 'inversion':
            # Inversion action (works when H = Z2, N is abelian)
            if h_idx == 0:  # identity in H
                return n_idx
            else:
                # Invert in N (works for cyclic groups)
                return (-n_idx) % n_size if n_idx != 0 else 0
        elif action_type == 'dihedral':
            # Dihedral-type action (creates dihedral groups when N=Zn, H=Z2)
            if h_idx == 0:
                return n_idx
            else:
                return (n_size - n_idx) % n_size
        else:
            # Default to trivial
            return n_idx

    # Create semi-direct product elements with simplified notation
    semidirect_elements = []
    for h_idx in range(h_size):
        for n_idx in range(n_size):
            h_elem = h_elements[h_idx]
            n_elem = n_elements[n_idx]
            # Simplify notation similar to product groups
            if h_elem == "e" and n_elem == "e":
                semidirect_elements.append("e")
            elif h_elem == "e":
                semidirect_elements.append(f"n{n_elem}")
            elif n_elem == "e":
                semidirect_elements.append(f"h{h_elem}")
            else:
                semidirect_elements.append(f"{n_elem}:{h_elem}")

    # Create semi-direct product table
    # (n1, h1) * (n2, h2) = (n1 * h1(n2), h1 * h2)
    semidirect_table = np.zeros((total_size, total_size), dtype=int)

    for i in range(total_size):
        for j in range(total_size):
            # Get indices in N and H
            h1_idx, n1_idx = divmod(i, n_size)
            h2_idx, n2_idx = divmod(j, n_size)

            # Apply the semi-direct product formula
            # Result in N: n1 * h1(n2)
            h1_n2 = get_action(h1_idx, n2_idx)
            result_n = n_table[n1_idx, h1_n2]

            # Result in H: h1 * h2
            result_h = h_table[h1_idx, h2_idx]

            # Convert back to product index
            semidirect_table[i, j] = result_h * n_size + result_n

    group_name = f"{normal_group.name} ⋊ {acting_group.name}"
    return GroupTable(semidirect_elements, semidirect_table, group_name)

def get_group_by_type(group_type):
    """Get a group instance by type string."""
    if group_type == 'Z2':
        return ExampleGroups.cyclic(2)
    elif group_type == 'Z3':
        return ExampleGroups.cyclic(3)
    elif group_type == 'Z4':
        return ExampleGroups.cyclic(4)
    elif group_type == 'Z5':
        return ExampleGroups.cyclic(5)
    elif group_type == 'Z6':
        return ExampleGroups.cyclic(6)
    elif group_type == 'Z7':
        return ExampleGroups.cyclic(7)
    elif group_type == 'Z8':
        return ExampleGroups.cyclic(8)
    elif group_type == 'Z9':
        return ExampleGroups.cyclic(9)
    elif group_type == 'Z10':
        return ExampleGroups.cyclic(10)
    elif group_type == 'Z11':
        return ExampleGroups.cyclic(11)
    elif group_type == 'Z12':
        return ExampleGroups.cyclic(12)
    elif group_type == 'Klein':
        return ExampleGroups.klein_four()
    elif group_type == 'S3':
        return create_s3()
    elif group_type == 'S4':
        return create_s4()
    elif group_type == 'A4':
        return create_a4()
    elif group_type == 'D3':
        return ExampleGroups.dihedral(3)
    elif group_type == 'D4':
        return ExampleGroups.dihedral(4)
    elif group_type == 'Q8':
        return ExampleGroups.quaternion()
    else:
        return None

# App layout
app.layout = html.Div([
    html.Div([
        html.H1('Interactive Cayley Tables', style={'textAlign': 'center', 'color': '#2c3e50'}),
        html.P('Explore group structure through colored visualizations',
               style={'textAlign': 'center', 'fontSize': '18px', 'color': '#7f8c8d'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),

    html.Div([
        html.Div([
            html.Label('Select Group Type:', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='group-type',
                options=[{'label': GROUPS[k]['name'], 'value': k} for k in GROUPS],
                value='cyclic',
                style={'marginBottom': '20px'}
            ),

            html.Div(id='parameter-container', children=[
                html.Label('Order:', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Slider(
                    id='group-order',
                    min=2,
                    max=12,
                    value=5,
                    marks={i: str(i) for i in range(2, 13)},
                    step=1,
                    tooltip={"placement": "bottom", "always_visible": False}
                )
            ], style={'marginBottom': '20px'}),

            # Hidden storage for product group selections and order
            dcc.Store(id='product-group-store', data={'group1': 'Z2', 'group2': 'Z3'}),
            dcc.Store(id='semidirect-store', data={'normal': 'Z3', 'acting': 'Z2', 'action': 'inversion'}),
            dcc.Store(id='order-store', data=5),

            html.Label('Color Scheme:', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.RadioItems(
                id='color-scheme',
                options=[{'label': COLOR_SCHEMES[k], 'value': k} for k in COLOR_SCHEMES],
                value='rainbow',
                style={'marginBottom': '10px'}
            ),

            html.Div([
                dcc.Checklist(
                    id='show-labels',
                    options=[{'label': ' Show element labels in cells', 'value': 'show'}],
                    value=['show'],
                    style={'marginBottom': '10px'}
                )
            ]),

            html.Label('Color Accessibility:', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='color-accessibility',
                options=[
                    {'label': 'Standard Colors', 'value': 'standard'},
                    {'label': 'Deuteranopia (Red-Green)', 'value': 'deuteranopia'},
                    {'label': 'Protanopia (Red-Green)', 'value': 'protanopia'},
                    {'label': 'Tritanopia (Blue-Yellow)', 'value': 'tritanopia'},
                    {'label': 'Universal High Contrast', 'value': 'universal'}
                ],
                value='standard',
                style={'marginBottom': '15px'}
            ),

            html.Div([
                html.Button('Show Analysis', id='analyze-btn', n_clicks=0,
                           style={'padding': '10px 20px', 'fontSize': '16px',
                                 'backgroundColor': '#3498db', 'color': 'white',
                                 'border': 'none', 'borderRadius': '5px',
                                 'cursor': 'pointer', 'width': '100%'})
            ], style={'marginBottom': '20px'}),

            html.Div(id='analysis-output', style={
                'backgroundColor': '#f8f9fa',
                'padding': '15px',
                'borderRadius': '5px',
                'maxHeight': '300px',
                'overflowY': 'auto',
                'marginBottom': '20px'
            }),

            html.Hr(style={'margin': '20px 0'}),

            html.Div([
                html.H4('Tips:', style={'color': '#2c3e50', 'marginBottom': '10px'}),
                html.Ul([
                    html.Li('Hover over cells to see group operations', style={'fontSize': '12px'}),
                    html.Li('Rainbow coloring shows overall structure', style={'fontSize': '12px'}),
                    html.Li('Order coloring reveals cyclic subgroups', style={'fontSize': '12px'}),
                    html.Li('Conjugacy coloring shows equivalence classes', style={'fontSize': '12px'}),
                    html.Li('Symmetric tables indicate abelian groups', style={'fontSize': '12px'})
                ], style={'paddingLeft': '20px', 'color': '#7f8c8d'})
            ])
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '15px',
                 'backgroundColor': '#ffffff', 'boxShadow': '2px 2px 10px rgba(0,0,0,0.1)',
                 'marginRight': '10px', 'borderRadius': '10px', 'height': '750px', 'overflowY': 'auto',
                 'verticalAlign': 'top'}),

        html.Div([
            dcc.Graph(id='cayley-table', style={'height': '550px'})
        ], style={'width': '350px', 'display': 'inline-block', 'backgroundColor': '#ffffff',
                 'padding': '10px', 'boxShadow': '2px 2px 10px rgba(0,0,0,0.1)',
                 'borderRadius': '10px', 'verticalAlign': 'top'})
    ], style={'margin': '10px', 'overflow': 'hidden', 'minHeight': '780px'}),

    html.Div(style={'clear': 'both'})
], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f5f6fa', 'minHeight': '100vh'})

@app.callback(
    [Output('parameter-container', 'children'),
     Output('product-group-store', 'data'),
     Output('semidirect-store', 'data'),
     Output('order-store', 'data')],
    [Input('group-type', 'value')],
    [State('product-group-store', 'data'),
     State('semidirect-store', 'data')]
)
def update_parameters(group_type, product_data, semidirect_data):
    """Update parameter controls based on selected group type."""
    if not group_type or group_type not in GROUPS:
        return [], product_data, semidirect_data, 5

    group_info = GROUPS[group_type]
    order_value = group_info.get('default', 5)

    if 'order' in group_info['params']:
        max_order = 12 if group_type == 'cyclic' else 8
        return [
            html.Label('Order:', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Slider(
                id='group-order',
                min=2,
                max=max_order,
                value=order_value,
                marks={i: str(i) for i in range(2, max_order + 1)},
                step=1,
                tooltip={"placement": "bottom", "always_visible": False}
            )
        ], product_data, semidirect_data, order_value
    elif group_type == 'product':
        # Product group selection
        return html.Div([
            html.Label('First Group (G):', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='product-group1',
                options=[{'label': PRODUCT_GROUPS[k], 'value': k} for k in PRODUCT_GROUPS],
                value=product_data.get('group1', 'Z2'),
                style={'marginBottom': '15px'}
            ),
            html.Label('Second Group (H):', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='product-group2',
                options=[{'label': PRODUCT_GROUPS[k], 'value': k} for k in PRODUCT_GROUPS],
                value=product_data.get('group2', 'Z3'),
                style={'marginBottom': '10px'}
            ),
            html.P('Creates the direct product G × H',
                  style={'fontStyle': 'italic', 'color': '#7f8c8d', 'fontSize': '12px'})
        ]), product_data, semidirect_data, order_value
    elif group_type == 'semidirect':
        # Semi-direct product selection
        return html.Div([
            html.Label('Normal Subgroup (N):', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='semidirect-normal',
                options=[{'label': PRODUCT_GROUPS[k], 'value': k} for k in PRODUCT_GROUPS],
                value=semidirect_data.get('normal', 'Z3'),
                style={'marginBottom': '15px'}
            ),
            html.Label('Acting Group (H):', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='semidirect-acting',
                options=[{'label': PRODUCT_GROUPS[k], 'value': k} for k in PRODUCT_GROUPS],
                value=semidirect_data.get('acting', 'Z2'),
                style={'marginBottom': '15px'}
            ),
            html.Label('Action Type:', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='semidirect-action',
                options=[{'label': SEMIDIRECT_ACTIONS[k], 'value': k} for k in SEMIDIRECT_ACTIONS],
                value=semidirect_data.get('action', 'inversion'),
                style={'marginBottom': '10px'}
            ),
            html.P('Creates the semi-direct product N ⋊ H',
                  style={'fontStyle': 'italic', 'color': '#7f8c8d', 'fontSize': '12px'})
        ]), product_data, semidirect_data, order_value
    else:
        # No parameters needed
        return html.Div([
            html.P(f"Fixed group: {group_info['name']}",
                  style={'fontStyle': 'italic', 'color': '#7f8c8d'})
        ]), product_data, semidirect_data, order_value

@app.callback(
    Output('product-group-store', 'data', allow_duplicate=True),
    [Input('product-group1', 'value'),
     Input('product-group2', 'value')],
    prevent_initial_call=True
)
def update_product_selection(group1, group2):
    """Update product group selection in storage."""
    return {'group1': group1, 'group2': group2}

@app.callback(
    Output('semidirect-store', 'data', allow_duplicate=True),
    [Input('semidirect-normal', 'value'),
     Input('semidirect-acting', 'value'),
     Input('semidirect-action', 'value')],
    prevent_initial_call=True
)
def update_semidirect_selection(normal, acting, action):
    """Update semi-direct group selection in storage."""
    return {'normal': normal, 'acting': acting, 'action': action}

@app.callback(
    Output('order-store', 'data', allow_duplicate=True),
    [Input('group-order', 'value')],
    prevent_initial_call=True
)
def update_order_store(order_value):
    """Update order in storage when slider changes."""
    return order_value if order_value else 5

@app.callback(
    Output('cayley-table', 'figure'),
    [Input('group-type', 'value'),
     Input('color-scheme', 'value'),
     Input('product-group-store', 'data'),
     Input('semidirect-store', 'data'),
     Input('order-store', 'data'),
     Input('show-labels', 'value'),
     Input('color-accessibility', 'value')])
def update_table(group_type, color_scheme, product_data, semidirect_data, order, show_labels, color_accessibility):
    """Update the Cayley table visualization."""
    if not group_type:
        return go.Figure()

    # Create the appropriate group
    if group_type == 'cyclic':
        group = ExampleGroups.cyclic(order if order else 5)
    elif group_type == 'dihedral':
        group = ExampleGroups.dihedral(order if order else 4)
    elif group_type == 'klein':
        group = ExampleGroups.klein_four()
    elif group_type == 'quaternion':
        group = ExampleGroups.quaternion()
    elif group_type == 's3':
        group = create_s3()
    elif group_type == 's4':
        group = create_s4()
    elif group_type == 'a4':
        group = create_a4()
    elif group_type == 'z2z2':
        group = create_z2z2()
    elif group_type == 'dicyclic':
        group = create_dicyclic(order if order else 3)
    elif group_type == 'product':
        group = create_product_group(product_data.get('group1', 'Z2'),
                                    product_data.get('group2', 'Z3'))
        if group is None:
            return go.Figure()
    elif group_type == 'semidirect':
        group = create_semidirect_product(semidirect_data.get('normal', 'Z3'),
                                        semidirect_data.get('acting', 'Z2'),
                                        semidirect_data.get('action', 'inversion'))
        if group is None:
            return go.Figure()
    else:
        return go.Figure()

    # Generate the figure
    # Check if labels should be shown (show_labels is a list from checklist)
    show_labels_bool = 'show' in show_labels if show_labels else False
    # Use color accessibility mode if provided
    accessibility = color_accessibility if color_accessibility else 'standard'
    fig = group.plot(color_scheme=color_scheme, show_labels=show_labels_bool,
                     color_accessibility=accessibility)

    # Update layout for better web display
    fig.update_layout(
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=12),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        )
    )

    return fig

@app.callback(
    Output('analysis-output', 'children'),
    [Input('analyze-btn', 'n_clicks')],
    [State('group-type', 'value'),
     State('product-group-store', 'data'),
     State('semidirect-store', 'data'),
     State('order-store', 'data')])
def show_analysis(n_clicks, group_type, product_data, semidirect_data, order):
    """Display group analysis."""
    if n_clicks == 0 or not group_type:
        return html.P("Click 'Show Analysis' to see group properties",
                     style={'color': '#7f8c8d', 'fontStyle': 'italic'})

    # Create the group
    if group_type == 'cyclic':
        group = ExampleGroups.cyclic(order if order else 5)
    elif group_type == 'dihedral':
        group = ExampleGroups.dihedral(order if order else 4)
    elif group_type == 'klein':
        group = ExampleGroups.klein_four()
    elif group_type == 'quaternion':
        group = ExampleGroups.quaternion()
    elif group_type == 's3':
        group = create_s3()
    elif group_type == 's4':
        group = create_s4()
    elif group_type == 'a4':
        group = create_a4()
    elif group_type == 'z2z2':
        group = create_z2z2()
    elif group_type == 'dicyclic':
        group = create_dicyclic(order if order else 3)
    elif group_type == 'product':
        group = create_product_group(product_data.get('group1', 'Z2'),
                                    product_data.get('group2', 'Z3'))
        if group is None:
            return html.P("Error creating product group")
    elif group_type == 'semidirect':
        group = create_semidirect_product(semidirect_data.get('normal', 'Z3'),
                                        semidirect_data.get('acting', 'Z2'),
                                        semidirect_data.get('action', 'inversion'))
        if group is None:
            return html.P("Error creating semi-direct product")
    else:
        return html.P("Select a group first")

    # Get analysis
    analysis = group.analyze()

    # Format element orders
    order_items = [html.Li(f"{elem}: {order}")
                  for elem, order in analysis['element_orders'].items()]

    # Format subgroups (limit display for large groups)
    subgroup_items = []
    for i, sg in enumerate(analysis['subgroups'][:10]):  # Show first 10 subgroups
        sg_str = '{' + ', '.join(sg) + '}'
        subgroup_items.append(html.Li(sg_str, style={'fontSize': '12px'}))
    if len(analysis['subgroups']) > 10:
        subgroup_items.append(html.Li(f"... and {len(analysis['subgroups']) - 10} more",
                                     style={'fontStyle': 'italic'}))

    # Format conjugacy classes
    conj_items = [html.Li('{' + ', '.join(cc) + '}', style={'fontSize': '12px'})
                 for cc in analysis['conjugacy_classes']]

    return html.Div([
        html.H4(f"{group.name} Analysis", style={'color': '#2c3e50', 'marginBottom': '15px'}),

        html.Div([
            html.Strong("Order: "),
            html.Span(f"{analysis['order']}")
        ], style={'marginBottom': '10px'}),

        html.Div([
            html.Strong("Identity: "),
            html.Span(f"{analysis['identity']}")
        ], style={'marginBottom': '10px'}),

        html.Div([
            html.Strong("Abelian: "),
            html.Span("Yes ✓" if analysis['is_abelian'] else "No ✗",
                     style={'color': '#27ae60' if analysis['is_abelian'] else '#e74c3c'})
        ], style={'marginBottom': '10px'}),

        html.Details([
            html.Summary("Element Orders", style={'fontWeight': 'bold', 'cursor': 'pointer',
                                                 'marginBottom': '5px'}),
            html.Ul(order_items, style={'marginTop': '5px', 'paddingLeft': '20px'})
        ], style={'marginBottom': '10px'}),

        html.Details([
            html.Summary(f"Subgroups ({len(analysis['subgroups'])})",
                        style={'fontWeight': 'bold', 'cursor': 'pointer', 'marginBottom': '5px'}),
            html.Ul(subgroup_items, style={'marginTop': '5px', 'paddingLeft': '20px'})
        ], style={'marginBottom': '10px'}),

        html.Details([
            html.Summary(f"Conjugacy Classes ({len(analysis['conjugacy_classes'])})",
                        style={'fontWeight': 'bold', 'cursor': 'pointer', 'marginBottom': '5px'}),
            html.Ul(conj_items, style={'marginTop': '5px', 'paddingLeft': '20px'})
        ])
    ])

# For deployment
server = app.server

if __name__ == '__main__':
    print("Starting Cayley Table Web App...")
    print("Open http://127.0.0.1:8050/ in your browser")
    app.run(debug=True, port=8050)