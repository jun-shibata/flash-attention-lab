import numpy as np

def attention_reference(q, k, v):

    """
    q: (Nq, d)
    k: (Nk, d)
    v: (Nk, dv)
    ret: (Nq, dv)
    """
  
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
  
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("q, k, v must be 2D arrays")
    if q.shape[1] != k.shape[1]:
        raise ValueError("q and k must have the same head dimension")
    if k.shape[0] != v.shape[0]:
        raise ValueError("k and v must have the same sequence length")
    if q.shape[1] == 0 or k.shape[0] == 0:
        raise ValueError("head dimension and key length must be positive")

    scores = (q @ k.T) / np.sqrt(q.shape[1])

    scores -= np.max(scores, axis=1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)

    return probabilities @ v


def make_inputs(nq=128, nk=256, d=64, dv=64, seed=0):
    rng = np.random.default_rng(seed)

    return tuple(
        rng.standard_normal(shape).astype(np.float32)
        for shape in ((nq, d), (nk, d), (nk, dv))                 
    )


def compare_output(actual, expected, atol=1e-5, rtol=1e-4):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)

    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {expected.shape}")

    if not np.all(np.isfinite(actual)):
        raise AssertionError("output contains NaN or Inf")

    error = actual - expected
    max_abs_error = np.max(np.abs(error))
    reference_norm = np.linalg.norm(expected)
    relative_l2_error = (
        np.linalg.norm(error) / reference_norm
        if reference_norm > 0
        else (0.0 if np.linalg.norm(error) == 0 else float("inf"))
    )

    print(f"max absolute error: {max_abs_error:.3e}")
    print(f"relative L2 error: {relative_l2_error:.3e}")

    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


if __name__ == "__main__":
    q, k, v = make_inputs()
    expected = attention_reference(q, k, v)

    np.savez("attention_case.npz", q=q, k=k, v=v, expected=expected,)

    print(f"reference output shape: {expected.shape}")