from tilelang.carver.arch.cuda import (
    CUDA,
    check_sm_version,
    is_ada_arch,
    is_ampere_arch,
    is_blackwell_arch,
    is_hopper_arch,
    is_tensorcore_supported_precision,
    is_volta_arch,
)


class _DummyCUDA(CUDA):
    """Minimal CUDA-like object for exercising family predicates without a GPU."""

    def __init__(self, sm_version: int):
        self.sm_version = sm_version
        self.target = None


def test_check_sm_version_handles_suffixes():
    assert check_sm_version("sm_80") == 80
    assert check_sm_version("sm_90") == 90
    assert check_sm_version("sm_90a") == 90
    assert check_sm_version("sm_100") == 100
    assert check_sm_version("sm_100f") == 100
    assert check_sm_version("sm_103a") == 103
    assert check_sm_version("sm_120") == 120
    assert check_sm_version("sm_120a") == 120
    assert check_sm_version("compute_90") == -1
    assert check_sm_version("invalid") == -1


def test_cuda_arch_family_classification():
    assert is_volta_arch(_DummyCUDA(70))
    assert is_volta_arch(_DummyCUDA(75))
    assert is_ampere_arch(_DummyCUDA(80))
    assert is_ada_arch(_DummyCUDA(89))
    assert is_hopper_arch(_DummyCUDA(90))
    assert is_hopper_arch(_DummyCUDA(99))
    assert is_blackwell_arch(_DummyCUDA(100))
    assert is_blackwell_arch(_DummyCUDA(120))
    assert not is_hopper_arch(_DummyCUDA(100))
    assert not is_blackwell_arch(_DummyCUDA(90))


def test_tensorcore_supported_precision_on_blackwell():
    # Blackwell supports at least the same precisions as Hopper.
    arch = _DummyCUDA(100)
    assert is_tensorcore_supported_precision("bfloat16", "float32", arch)
    assert is_tensorcore_supported_precision("float16", "float16", arch)
    assert is_tensorcore_supported_precision("float8_e4m3", "float32", arch)


def test_tensorcore_supported_precision_falls_back_for_future_arch():
    arch = _DummyCUDA(130)
    assert is_tensorcore_supported_precision("bfloat16", "float32", arch)
    assert is_tensorcore_supported_precision("float8_e4m3", "float32", arch)


if __name__ == "__main__":
    test_check_sm_version_handles_suffixes()
    test_cuda_arch_family_classification()
    test_tensorcore_supported_precision_on_blackwell()
    test_tensorcore_supported_precision_falls_back_for_future_arch()
