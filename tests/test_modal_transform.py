"""CPU modal/autograd/geometry checks. Source prepared; not executed by agent."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from mpi4py import MPI
from pysemtools.datatypes import Mesh, Coef
from pysemtools.datatypes.coef import get_transform_matrix, apply_1d_operators_torch
from pysemtools.compression.legendre_truncation_bitplaneV4 import DiscreetLegendreTruncationBPWeightedV4


def mesh(n=4, dtype=np.float64):
    _, _, _, nodes, _ = get_transform_matrix(n, 3, apply_1d_operators=True)
    z, y, x = np.meshgrid(nodes, nodes, nodes, indexing='ij')
    return Mesh(MPI.COMM_SELF, x=np.stack((x, 2*x+5)).astype(dtype),
                y=np.stack((y, y)).astype(dtype), z=np.stack((z, z)).astype(dtype),
                create_connectivity=False)


@pytest.mark.parametrize('n', [3, 4])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_modal_order_and_inverse(n, dtype):
    v, vinv, _, nodes, _ = get_transform_matrix(n, 3, apply_1d_operators=True)
    # Independent analytic polynomial samples, including the top GLL mode.
    indices = (n-1, 1, 0)  # k, j, i
    one_d = []
    for degree in indices:
        coefficients = np.eye(n)[degree]
        h = 2/(n-1) if degree == n-1 else 2/(2*degree+1)
        one_d.append(np.polynomial.legendre.legval(nodes, coefficients)/np.sqrt(h))
    field = np.einsum('k,j,i->kji', *one_d)
    field = torch.tensor(field, dtype=dtype)[None, None].expand(2, 3, n, n, n)
    geometry = mesh(n, dtype=np.float32 if dtype == torch.float32 else np.float64)
    coef = Coef(geometry, MPI.COMM_SELF, bckend='torch', device='cpu')
    result = coef.nodal_to_modal(field)
    expected = torch.zeros_like(result)
    expected[..., indices[0], indices[1], indices[2]] = 1.
    tolerance = 3.e-5 if dtype == torch.float32 else 1.e-10
    torch.testing.assert_close(result, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(coef.modal_to_nodal(result),
                               field, atol=tolerance, rtol=tolerance)


def test_tensor_product_matches_full_numpy_operator():
    v, vinv, _, _, _ = get_transform_matrix(4, 3, apply_1d_operators=True)
    values = np.arange(2*3*4**3, dtype=float).reshape(2, 3, 4, 4, 4)
    full = np.kron(vinv, np.kron(vinv, vinv))
    expected = (values.reshape(-1, 4**3) @ full.T).reshape(values.shape)
    coef = Coef(mesh(), MPI.COMM_SELF)
    np.testing.assert_allclose(coef.nodal_to_modal(values), expected, atol=1.e-10)


def test_torch_coef_cpu_and_prediction_gradients():
    geometry = mesh()
    coef = Coef(geometry, MPI.COMM_SELF, bckend='torch', device='cpu')
    field = torch.tensor(2*geometry.x + 3*geometry.y + 4*geometry.z,
                         dtype=torch.float64, requires_grad=True)
    coefficients = coef.nodal_to_modal(field)
    torch.testing.assert_close(coef.modal_to_nodal(coefficients), field, atol=1.e-10, rtol=1.e-10)
    derivative = coef.dudxyz(field, coef.drdx, coef.dsdx, coef.dtdx)
    torch.testing.assert_close(derivative, torch.full_like(derivative, 2.), atol=1.e-10, rtol=1.e-10)
    (coefficients.square().sum()+derivative.square().sum()).backward()
    assert field.grad is not None and torch.isfinite(field.grad).all() and field.grad.abs().sum() > 0


def test_noncontiguous_field_and_operator_autograd():
    values = torch.arange(2*4**3, dtype=torch.float64).reshape(2, 4, 4, 4).requires_grad_()
    field = values.transpose(-1, -2)
    op = torch.eye(4, dtype=torch.float64, requires_grad=True)
    matrix = op.reshape(1, 1, 4, 4)
    result = apply_1d_operators_torch(field.reshape(-1, 1, 4**3, 1),
                                     matrix, matrix, matrix).reshape(field.shape)
    torch.testing.assert_close(result, field)
    result.square().sum().backward()
    assert values.grad is not None and op.grad is not None
    coef = Coef(mesh(), MPI.COMM_SELF, bckend='torch', device='cpu')
    assert torch.isfinite(coef.dudrst(field, direction='r')).all()


def test_v4_geometry_diagonal_and_full_gradient_are_distinct():
    geometry = mesh()
    coef = Coef(geometry, MPI.COMM_SELF, store_multidimensional_operators=True)
    v4 = DiscreetLegendreTruncationBPWeightedV4(comm=MPI.COMM_SELF, msh=geometry, coef=coef)
    diagonal = v4.precompute_gram_diagonal('gradient')
    np.testing.assert_allclose(diagonal[1, 0, 0, 1], .5*diagonal[0, 0, 0, 1], rtol=1.e-10)
    np.testing.assert_allclose(diagonal[:, 0, 0, 0], 0., atol=1.e-20)
    L = coef.v_xd
    gram = np.zeros((L.shape[1], L.shape[1]))
    for direction in ('x', 'y', 'z'):
        K = (getattr(coef, 'drd'+direction)[0].reshape(-1, 1) * (coef.dr_xd @ L)
             + getattr(coef, 'dsd'+direction)[0].reshape(-1, 1) * (coef.ds_xd @ L)
             + getattr(coef, 'dtd'+direction)[0].reshape(-1, 1) * (coef.dt_xd @ L))
        gram += K.T @ (coef.B[0].reshape(-1, 1) * K)
    np.testing.assert_allclose(diagonal[0].reshape(-1), np.diag(gram), atol=1.e-10)
    error = np.zeros(L.shape[1])
    error[1], error[3] = 1., 1.
    assert not np.isclose(error @ gram @ error, np.dot(np.diag(gram), error**2))
