import pytest
from smolagents.local_python_executor import InterpreterError, LocalPythonExecutor

from agents.base import block_numpy_file_io, expand_authorized_imports


@pytest.fixture
def numpy_executor():
    block_numpy_file_io()
    executor = LocalPythonExecutor(additional_authorized_imports=expand_authorized_imports(["numpy"]))
    executor.send_tools({})
    return executor


def test_expansion_lists_safe_numpy_submodules_only():
    modules = expand_authorized_imports(["numpy"])
    assert "numpy" in modules and "numpy.linalg" in modules
    assert not any(m in modules for m in ("numpy.f2py", "numpy.testing", "numpy.*"))
    assert expand_authorized_imports(None) == []


def test_numpy_computation_runs(numpy_executor):
    out = numpy_executor("import numpy as np\nnp.linalg.inv(np.array([[2.0, 0.0], [0.0, 4.0]]))[1, 1]")
    assert out.output == pytest.approx(0.25)


@pytest.mark.parametrize("call", ["np.load('{p}')", "np.savetxt('{p}', np.zeros(2))", "np.fromfile('{p}')"])
def test_numpy_file_io_is_blocked(numpy_executor, call, tmp_path):
    target = tmp_path / "blocked.dat"
    with pytest.raises(InterpreterError, match="disabled"):
        numpy_executor("import numpy as np\n" + call.format(p=target))
    assert not target.exists()


def test_unlisted_numpy_submodule_is_blocked(numpy_executor):
    with pytest.raises(InterpreterError, match="Forbidden"):
        numpy_executor("import numpy as np\nnp.f2py")


def test_numpy_not_importable_by_default():
    executor = LocalPythonExecutor(additional_authorized_imports=[])
    executor.send_tools({})
    with pytest.raises(InterpreterError):
        executor("import numpy")
