from __future__ import annotations

from typing import Callable, Literal, TYPE_CHECKING
if TYPE_CHECKING:
    import ase

import numpy as np
import torch

from graph_pes.atomic_graph import AtomicGraph, is_batch
from graph_pes.utils.misc import random_rotation_matrix

Transform = Callable[[AtomicGraph], AtomicGraph]
TransformName = Literal["identity", "random_rotation"]


def identity_transform(graph: AtomicGraph) -> AtomicGraph:
    """A transform that returns the graph unchanged."""
    return graph


def random_rotation(graph: AtomicGraph) -> AtomicGraph:
    """Randomly rotate the atom positions, cell and forces (if present)."""

    if is_batch(graph):
        raise ValueError("Batched graphs are not supported")

    R = random_rotation_matrix()

    props = {**graph.properties}
    if "forces" in props:
        props["forces"] = props["forces"] @ R

    new_graph = graph._replace(
        R=graph.R @ R,
        cell=graph.cell @ R,
        properties=props,
    )
    return new_graph


def parse_transform(t: TransformName | Transform | None) -> Transform:
    if t is None or t == "identity":
        return identity_transform
    elif t == "random_rotation":
        return random_rotation
    elif callable(t):
        return t
    else:
        raise ValueError(f"Unknown transform: {t}")

_KEYWORD_TO_IRREPS: dict[str, str] = {
    "isotropic": "0e",
    "symmetric": "0e+2e",
    "antisymmetric": "1e",
    "full": "0e+1e+2e",
}


class CartesianTensorConverter:
    r"""
    Converts per-atom rank-2 tensors between their Cartesian (3x3)
    and irreducible spherical representations.

    Parameters
    ----------
    irreps
        Which irreducible components to use. Accepts either a keyword
        or e3nn irreps notation:

        - ``"isotropic"`` / ``"0e"``          (1 component)
        - ``"antisymmetric"`` / ``"1e"``      (3 components)
        - ``"symmetric"`` / ``"0e+2e"``       (6 components)
        - ``"full"`` / ``"0e+1e+2e"``         (9 components)
        - any valid subset, e.g. ``"2e"``     (5 components)

    Attributes
    ----------
    n_components : int
        Number of spherical components produced by `to_spherical`.
    irreps : e3nn.o3.Irreps
        The irreducible representations in the output.

    Examples
    --------
    Magnetic shielding — full asymmetric tensor (9 components):

    .. code-block:: python

        converter_ms = CartesianTensorConverter("full")
        sph = converter_ms.to_spherical(T)  # shape (..., 9)

    Electric field gradient — traceless symmetric tensor (5 components):

    .. code-block:: python

        converter_efg = CartesianTensorConverter("2e")
        sph = converter_efg.to_spherical(T)  # shape (..., 5)

    Pre-process an :class:`ase.Atoms` trajectory before training:

    .. code-block:: python

        import ase.io
        from graph_pes.data.transforms import CartesianTensorConverter

        converter = CartesianTensorConverter("2e")
        structures = ase.io.read("trajectory.xyz", index=":")
        for atoms in structures:
            converter.atoms_to_spherical(atoms, key="efg")
        ase.io.write("trajectory_sph.xyz", structures)
    """

    def __init__(self, irreps: str = "full"):
        from e3nn import o3

        irreps_str = _KEYWORD_TO_IRREPS.get(irreps, irreps)

        rtp_symm = o3.ReducedTensorProducts("ij=ji", i="1o", j="1o")
        rtp_anti = o3.ReducedTensorProducts("ij=-ji", i="1o", j="1o")

        Q_symm = rtp_symm.change_of_basis.flatten(-2)  # (n_symm, 9)
        Q_anti = rtp_anti.change_of_basis.flatten(-2)   # (n_anti, 9)

        symm_list = list(rtp_symm.irreps_out)
        anti_list = list(rtp_anti.irreps_out)
        first_dim = symm_list[0].mul * symm_list[0].ir.dim
        Q_full = torch.cat(
            [Q_symm[:first_dim], Q_anti, Q_symm[first_dim:]], dim=0
        )  # (9, 9)

        # map irrep label → row indices in Q_full
        irrep_to_rows: dict[str, list[int]] = {}
        row = 0
        for mul_ir in [symm_list[0]] + anti_list + symm_list[1:]:
            dim = mul_ir.mul * mul_ir.ir.dim
            key = str(mul_ir.ir)
            irrep_to_rows.setdefault(key, []).extend(range(row, row + dim))
            row += dim

        requested_irreps = o3.Irreps(irreps_str)
        selected: list[int] = []
        for mul_ir in requested_irreps:
            ir_str = str(mul_ir.ir)
            if ir_str not in irrep_to_rows:
                raise ValueError(
                    f"Irrep '{ir_str}' is not valid for a rank-2 tensor. "
                    f"Available: {sorted(irrep_to_rows)}"
                )
            if mul_ir.mul != 1:
                raise ValueError(
                    f"Multiplicity must be 1 for each irrep; "
                    f"got {mul_ir.mul}x{ir_str}"
                )
            selected.extend(irrep_to_rows[ir_str])

        self._Q = Q_full[selected]  # (n_components, 9)
        self.irreps = o3.Irreps(irreps_str)
        self.n_components: int = len(selected)

    def to_spherical(self, cartesian: torch.Tensor) -> torch.Tensor:
        """
        Convert per-atom Cartesian tensors to their irreducible spherical
        representation.

        Parameters
        ----------
        cartesian
            Per-atom Cartesian tensors of shape (..., 3, 3).

        Returns
        -------
        torch.Tensor
            Spherical representation of shape (..., n_components).
        """
        Q = self._Q.to(device=cartesian.device, dtype=cartesian.dtype)
        return cartesian.flatten(-2) @ Q.T

    def to_cartesian(self, spherical: torch.Tensor) -> torch.Tensor:
        """
        Convert per-atom spherical tensors back to Cartesian representation.

        When only a subset of irreps is used, the result is the
        projection of the full tensor onto those components only.

        Parameters
        ----------
        spherical
            Per-atom spherical tensors of shape (..., n_components).

        Returns
        -------
        torch.Tensor
            Cartesian tensors of shape (..., 3, 3).
        """
        Q = self._Q.to(device=spherical.device, dtype=spherical.dtype)
        return (spherical @ Q).view(*spherical.shape[:-1], 3, 3)

    def atoms_to_spherical(
        self, atoms: ase.Atoms, key: str, out_key: str | None = None
    ) -> None:
        """
        Convert per-atom Cartesian tensor from an ase.Atoms object
        to spherical basis and store it under a new key.

        Parameters
        ----------
        atoms
            ase.Atoms object to modify in-place.
        key
            Key in atoms.arrays holding the Cartesian tensor,
            expected shape (N_atoms, 3, 3) or (N_atoms, 9).
        out_key
            Key to store the spherical representation.
            Defaults to ``key + "_sph"``.
        """
        if out_key is None:
            out_key = f"{key}_sph"

        raw = atoms.arrays[key]
        T = torch.from_numpy(raw.reshape(-1, 3, 3))
        sph = self.to_spherical(T)
        atoms.arrays[out_key] = sph.numpy()

    def atoms_to_cartesian(
        self, atoms: ase.Atoms, key: str, out_key: str | None = None
    ) -> None:
        """
        Convert per-atom spherical tensor from a ase.Atoms 
        object back to Cartesian basis

        Parameters
        ----------
        atoms
           ase.Atoms object.
        key
            Key in atoms.arrays` holding the spherical tensor,
            with shape (N_atoms, 9).
        out_key
            Key to store the Cartesian tensor.
            Defaults to key + "_cart".
        """
        if out_key is None:
            out_key = f"{key}_cart"

        raw = atoms.arrays[key]
        sph = torch.from_numpy(raw.reshape(-1, self.n_components))
        cart = self.to_cartesian(sph)
        atoms.arrays[out_key] = cart.numpy().reshape(-1, 9)
