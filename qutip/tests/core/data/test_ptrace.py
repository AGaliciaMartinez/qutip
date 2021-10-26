from .test_mathematics import UnaryOpMixin, shapes_unary
import numpy as np
import scipy as sc
import pytest
from qutip import data
from qutip.core.data import CSR, Dense
import numbers

class TestTrace(UnaryOpMixin):
    def op_numpy(self, matrix, dim, ):
        return np.sum(np.diag(matrix))

    # I purposely do not use shapes_square to ensure that dims will always
    # match the matrix dimensions.
    shapes = [
        (pytest.param((2*3*4*2, 2*3*4*2), id="dim_[2,3,4,2]"),),
    ]
    dim = [2, 3, 4, 2]
    bad_shapes = [
        (x,) for x in shapes_unary() if x.values[0][0] != x.values[0][1]
    ]
    specialisations = [
        pytest.param(data.ptrace_csr, CSR, CSR),
    ]

    # Trace actually does have bad shape, so we put that in too.
    def test_incorrect_shape_raises(self, op, data_m, dim, sel):
        """
        Test that the operation produces a suitable error if the shape is not a
        square matrix.
        """
        with pytest.raises(ValueError):
            op(data_m())


    @pytest.mark.parametrize('sel', [[0], [0, 1], [1, 0] , [1, 3, 2]])
    def test_mathematically_correct(self, op, data_m, out_type, sel):
        matrix = data_m()
        expected = self.op_numpy(matrix.to_array(), dim, sel)
        test = op(matrix, dim, sel)
        assert isinstance(test, out_type)
        if issubclass(out_type, Data):
            assert test.shape == expected.shape
            np.testing.assert_allclose(test.to_array(), expected,
                                       atol=self.tol)
        elif out_type is list:
            for test_, expected_ in zip(test, expected):
                assert test_.shape == expected_.shape
                np.testing.assert_allclose(test_.to_array(),
                                           expected_, atol=self.tol)
        else:
            assert abs(test - expected) < self.tol
