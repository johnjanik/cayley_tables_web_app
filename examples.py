#!/usr/bin/env python3
"""Examples demonstrating the Cayley table visualizations."""

from cayley_tables import GroupTable, ExampleGroups, compare_groups, interactive_explorer
import numpy as np


def main():
    """Run various group visualization examples."""

    # Example 1: Cyclic Groups
    print("=" * 50)
    print("Example 1: Cyclic Groups")
    print("=" * 50)

    c3 = ExampleGroups.cyclic(3)
    c4 = ExampleGroups.cyclic(4)
    c5 = ExampleGroups.cyclic(5)

    # Single group visualization
    fig = c5.plot(color_scheme='rainbow', title="Cyclic Group C_5")
    fig.show()

    # Compare cyclic groups
    fig = compare_groups(c3, c4, c5, color_scheme='order')
    fig.show()

    # Example 2: Dihedral Groups
    print("\n" + "=" * 50)
    print("Example 2: Dihedral Groups")
    print("=" * 50)

    d3 = ExampleGroups.dihedral(3)  # Triangle symmetries
    d4 = ExampleGroups.dihedral(4)  # Square symmetries

    # Visualize with different color schemes
    fig = d4.plot(color_scheme='conjugacy', title="D_4 - Colored by Conjugacy Classes")
    fig.show()

    # Example 3: Klein Four-Group
    print("\n" + "=" * 50)
    print("Example 3: Klein Four-Group")
    print("=" * 50)

    klein = ExampleGroups.klein_four()
    analysis = klein.analyze()

    print(f"Klein Four-Group Properties:")
    print(f"  Abelian: {analysis['is_abelian']}")
    print(f"  All non-identity elements have order 2")

    fig = klein.plot(color_scheme='order')
    fig.show()

    # Example 4: Quaternion Group
    print("\n" + "=" * 50)
    print("Example 4: Quaternion Group Q_8")
    print("=" * 50)

    q8 = ExampleGroups.quaternion()

    # Interactive explorer showing all color schemes
    interactive_explorer(q8)

    # Example 5: Custom Symmetric Group S_3
    print("\n" + "=" * 50)
    print("Example 5: Symmetric Group S_3")
    print("=" * 50)

    # S_3 permutations
    s3_elements = ['e', '(12)', '(13)', '(23)', '(123)', '(132)']
    s3_table = np.array([
        [0, 1, 2, 3, 4, 5],  # e
        [1, 0, 4, 5, 2, 3],  # (12)
        [2, 5, 0, 4, 3, 1],  # (13)
        [3, 4, 5, 0, 1, 2],  # (23)
        [4, 3, 1, 2, 5, 0],  # (123)
        [5, 2, 3, 1, 0, 4]   # (132)
    ])

    s3 = GroupTable(s3_elements, s3_table, "S_3")

    # Find and highlight subgroups
    subgroups = s3.find_subgroups()
    print(f"S_3 has {len(subgroups)} subgroups:")
    for sg_indices in subgroups:
        sg_elements = [s3.elements[i] for i in sg_indices]
        print(f"  {{{', '.join(sg_elements)}}}")

    # Highlight the alternating group A_3 (index 3 subgroup)
    a3_indices = [0, 4, 5]  # {e, (123), (132)}
    fig = s3.plot(color_scheme='rainbow', highlight_subgroup=a3_indices,
                  title="S_3 with A_3 subgroup highlighted")
    fig.show()

    # Example 6: Modular Arithmetic Group
    print("\n" + "=" * 50)
    print("Example 6: Z_6 under addition mod 6")
    print("=" * 50)

    z6_elements = [str(i) for i in range(6)]
    z6_table = np.array([[(i + j) % 6 for j in range(6)] for i in range(6)])
    z6 = GroupTable(z6_elements, z6_table, "Z_6")

    fig = z6.plot(color_scheme='order', title="Z_6 - Colored by element order")
    fig.show()

    # Example 7: Direct Product
    print("\n" + "=" * 50)
    print("Example 7: Direct Product Z_2 × Z_2 (isomorphic to Klein four)")
    print("=" * 50)

    z2z2_elements = ['(0,0)', '(0,1)', '(1,0)', '(1,1)']
    z2z2_table = np.array([
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0]
    ])
    z2z2 = GroupTable(z2z2_elements, z2z2_table, "Z_2 × Z_2")

    # Compare with Klein four to show isomorphism
    fig = compare_groups(klein, z2z2, color_scheme='rainbow')
    fig.show()

    print("\nNotice that Z_2 × Z_2 and Klein four-group have the same structure!")

    # Example 8: Non-abelian vs Abelian groups of same order
    print("\n" + "=" * 50)
    print("Example 8: Comparing S_3 (non-abelian) with Z_6 (abelian)")
    print("=" * 50)

    fig = compare_groups(s3, z6, color_scheme='rainbow')
    fig.show()

    print("\nNotice the asymmetry in S_3 (non-abelian) vs symmetry in Z_6 (abelian)")


def coset_visualization_example():
    """Demonstrate coset visualization."""
    print("\n" + "=" * 50)
    print("Coset Visualization Example")
    print("=" * 50)

    # Create D_4 (symmetries of a square)
    d4 = ExampleGroups.dihedral(4)

    # Find subgroups
    subgroups = d4.find_subgroups()

    # Pick a non-trivial subgroup (e.g., rotations only)
    rotation_subgroup = [0, 1, 2, 3]  # {e, r, r^2, r^3}

    print(f"D_4 with rotation subgroup highlighted:")
    fig = d4.plot(color_scheme='rainbow', highlight_subgroup=rotation_subgroup,
                  title="D_4 - Rotation subgroup highlighted")
    fig.show()

    # Left cosets visualization
    print("\nLeft cosets of the rotation subgroup:")
    # First coset: the subgroup itself
    print(f"  Coset 1: {{{', '.join([d4.elements[i] for i in rotation_subgroup])}}}")
    # Second coset: reflection * subgroup
    reflection_coset = [4, 5, 6, 7]
    print(f"  Coset 2: {{{', '.join([d4.elements[i] for i in reflection_coset])}}}")


def lagrange_theorem_demo():
    """Demonstrate Lagrange's theorem visually."""
    print("\n" + "=" * 50)
    print("Lagrange's Theorem Demonstration")
    print("=" * 50)

    groups = [
        ExampleGroups.cyclic(6),
        ExampleGroups.dihedral(3),
        ExampleGroups.quaternion()
    ]

    for group in groups:
        analysis = group.analyze()
        print(f"\n{group.name} (order {analysis['order']}):")

        subgroups = group.find_subgroups()
        for sg_indices in subgroups:
            sg_order = len(sg_indices)
            if sg_order > 1 and sg_order < group.n:  # Non-trivial subgroups
                sg_elements = [group.elements[i] for i in sg_indices]
                print(f"  Subgroup {{{', '.join(sg_elements[:3])}{'...' if len(sg_elements) > 3 else ''}}} "
                      f"has order {sg_order}")
                print(f"    {analysis['order']} ÷ {sg_order} = {analysis['order'] // sg_order} "
                      f"(Lagrange's theorem: subgroup order divides group order)")

                # Visualize with subgroup highlighted
                fig = group.plot(highlight_subgroup=sg_indices,
                                title=f"{group.name} - Subgroup of order {sg_order}")
                fig.show()
                break  # Show just one example per group


if __name__ == "__main__":
    print("Cayley Table Visualization Examples\n")
    print("This script demonstrates various group visualizations using colored Cayley tables.")
    print("Each color scheme reveals different structural properties of the groups.\n")

    # Run main examples
    main()

    # Additional specialized examples
    print("\n" + "=" * 50)
    print("Additional Examples")
    print("=" * 50)

    coset_visualization_example()
    lagrange_theorem_demo()

    print("\n" + "=" * 50)
    print("Examples complete! The visualizations show how colors can reveal:")
    print("  - Group structure and patterns")
    print("  - Element orders")
    print("  - Conjugacy classes")
    print("  - Subgroup structure")
    print("  - Symmetries (abelian groups) vs asymmetries (non-abelian)")
    print("=" * 50)