#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
import graphdot.linalg.low_rank as lr


def test_LATR():
    N = 100
    k = 10
    r = 5
    lhs = np.random.randn(N, k)
    rhs = np.random.randn(k, N)
    v = np.random.randn(N)
    w = np.random.randn(N, r)
    A = lr.LATR(lhs, rhs)
    assert(repr(lhs.shape) in repr(A))
    assert(repr(rhs.shape) in repr(A))
    assert(A.lhs is lhs)
    assert(A.rhs is rhs)
    assert(np.allclose(A.todense(), lhs @ rhs))
    assert(np.allclose(A.T.todense(), A.todense().T))
    assert(np.allclose(-A.todense(), -(A.todense())))
    assert(np.allclose(A.diagonal(), A.todense().diagonal()))
    assert(np.allclose(A.trace(), np.trace(A.todense())))
    assert(A.quadratic(v, v) == pytest.approx(v @ A.todense() @ v))
    assert(A.quadratic(v, v) == pytest.approx(v @ A.todense() @ v))
    assert(np.allclose(
        A.quadratic_diag(w.T, w),
        (w.T @ A.todense() @ w).diagonal()
    ))


def test_LLT():
    N = 100
    k = 10
    r = 5
    lhs = np.random.randn(N, k)
    v = np.random.randn(N)
    w = np.random.randn(N, r)
    A = lr.LLT(lhs)
    assert(repr(lhs.shape) in repr(A))
    assert(repr(lhs.T.shape) in repr(A))
    assert(np.allclose(A.todense(), lhs @ lhs.T))
    assert(np.allclose(A.T.todense(), A.todense().T))
    assert(np.allclose(-A.todense(), -(A.todense())))
    assert(np.allclose(A.diagonal(), A.todense().diagonal()))
    assert(np.allclose(A.trace(), np.trace(A.todense())))
    assert(A.quadratic(v, v) == pytest.approx(v @ A.todense() @ v))
    assert(A.quadratic(v, v) == pytest.approx(v @ A.todense() @ v))
    assert(np.allclose(
        A.quadratic_diag(w.T, w),
        (w.T @ A.todense() @ w).diagonal()
    ))
    assert(np.allclose((A @ A.pinv() @ A).todense(), A.todense()))
    assert(A.logdet() == pytest.approx(
        np.prod(np.linalg.slogdet(lhs.T @ lhs))
    ))
    assert(A.cond() == pytest.approx(np.linalg.cond(lhs.T @ lhs)))
    assert(np.allclose(
        (A**2).todense(),
        np.linalg.matrix_power(A.todense(), 2)
    ))


@pytest.mark.parametrize('X', [
    lr.LATR(np.random.randn(100, 10), np.random.randn(10, 100)),
    lr.LLT(np.random.randn(100, 10)),
])
@pytest.mark.parametrize('Y', [
    lr.LATR(np.random.randn(100, 10), np.random.randn(10, 100)),
    lr.LLT(np.random.randn(100, 10)),
])
def test_sum(X, Y):
    A = X + Y
    v = np.random.randn(100)
    assert(np.allclose(A.todense(), X.todense() + Y.todense()))
    assert(np.allclose(A.T.todense(), A.todense().T))
    assert(np.allclose(-A.todense(), -(A.todense())))
    assert(np.allclose(A.diagonal(), A.todense().diagonal()))
    assert(np.allclose(A.trace(), np.trace(A.todense())))
    assert(A.quadratic(v, v) == pytest.approx(v @ A.todense() @ v))
    assert(A.quadratic(v, v) == pytest.approx(v @ A.todense() @ v))


@pytest.mark.parametrize('X', [
    lr.LATR(np.random.randn(100, 10), np.random.randn(10, 100)),
    lr.LLT(np.random.randn(100, 10)),
])
@pytest.mark.parametrize('Y', [
    lr.LATR(np.random.randn(100, 10), np.random.randn(10, 100)),
    lr.LLT(np.random.randn(100, 10)),
])
def test_sub(X, Y):
    A = X - Y
    v = np.random.randn(100)
    assert(np.allclose(A.todense(), X.todense() - Y.todense()))
    assert(np.allclose(A.T.todense(), A.todense().T))
    assert(np.allclose(-A.todense(), -(A.todense())))
    assert(np.allclose(A.diagonal(), A.todense().diagonal()))
    assert(np.allclose(A.trace(), np.trace(A.todense())))
    assert(A.quadratic(v, v) == pytest.approx(v @ A.todense() @ v))
    assert(A.quadratic(v, v) == pytest.approx(v @ A.todense() @ v))


@pytest.mark.parametrize('X', [
    lr.LATR(np.random.randn(100, 10), np.random.randn(10, 100)),
    lr.LLT(np.random.randn(100, 10)),
    lr.LLT(np.random.randn(100, 10)) + lr.LLT(np.random.randn(100, 10)),
    lr.LLT(np.random.randn(100, 10)) - lr.LLT(np.random.randn(100, 10)),
])
@pytest.mark.parametrize('Y', [
    lr.LATR(np.random.randn(100, 10), np.random.randn(10, 100)),
    lr.LLT(np.random.randn(100, 10)),
    lr.LLT(np.random.randn(100, 10)) + lr.LLT(np.random.randn(100, 10)),
    lr.LLT(np.random.randn(100, 10)) - lr.LLT(np.random.randn(100, 10)),
])
def test_mul(X, Y):
    A = X @ Y
    v = np.random.randn(100)
    assert(np.allclose(A.todense(), X.todense() @ Y.todense()))
    assert(np.allclose(A.T.todense(), A.todense().T))
    assert(np.allclose(-A.todense(), -(A.todense())))
    assert(np.allclose(A.diagonal(), A.todense().diagonal()))
    assert(np.allclose(A.trace(), np.trace(A.todense())))
    assert(A.quadratic(v, v) == pytest.approx(v @ A.todense() @ v))
    assert(A.quadratic(v, v) == pytest.approx(v @ A.todense() @ v))


def test_long_sum():
    X = [lr.LLT(np.random.randn(100, 10)) for _ in range(10)]
    A = X[0]
    for x in X[1:]:
        A = A + x
    v = np.random.randn(100)
    assert(np.allclose(A.todense(), np.sum([x.todense() for x in X], axis=0)))
    assert(np.allclose(A.T.todense(), A.todense().T))
    assert(np.allclose(-A.todense(), -(A.todense())))
    assert(np.allclose(A.diagonal(), A.todense().diagonal()))
    assert(np.allclose(A.trace(), np.trace(A.todense())))
    assert(A.quadratic(v, v) == pytest.approx(v @ A.todense() @ v))
    assert(A.quadratic(v, v) == pytest.approx(v @ A.todense() @ v))


def test_dot():
    X = np.random.randn(100, 10)
    Y = np.random.randn(100, 10)
    assert(isinstance(lr.dot(X, Y.T), lr.LATR))
    assert(isinstance(lr.dot(X, X.T), lr.LATR))
    assert(isinstance(lr.dot(X), lr.LLT))


@pytest.mark.parametrize('lambd', [-5, -4, -3, -2, -1, 0])
@pytest.mark.parametrize('r', [1, 3, 5, 10, 20, 30])
def test_pinvh(lambd, r):
    N = 100
    k = 10
    rcond = 1e-10
    lhs = np.random.randn(N, k)
    for _ in range(5):
        A = lr.LLT(lhs)
        d = np.zeros(N)
        d[np.random.choice(N, r, False)] = np.random.lognormal(lambd, 1)
        Ainv = lr.pinvh(A, d, rcond=rcond)
        Ad = A.todense()
        Ad.flat[::len(Ad) + 1] += d
        assert(np.allclose(
            Ainv.todense(),
            np.linalg.pinv(Ad, rcond=rcond),
            atol=1e-6, rtol=1e-2
        ))
