import pytest
from blinkmimic.deploy.core import inference_device

@pytest.mark.parametrize("value,expected",[("cpu","cpu"),("cuda:0","cuda:0"),("cuda:02","cuda:2")])
def test_device(value,expected):
    assert inference_device(value)==expected

@pytest.mark.parametrize("value",["cuda","gpu","cuda:-1","cpu:0","cuda:abc"])
def test_invalid_device(value):
    with pytest.raises(ValueError):inference_device(value)

def test_explicit_cuda_requires_provider(monkeypatch):
    import sys
    from types import SimpleNamespace
    from blinkmimic.deploy.core import OnnxPolicy
    monkeypatch.setitem(sys.modules,"onnxruntime",SimpleNamespace(get_available_providers=lambda:["CPUExecutionProvider"]))
    with pytest.raises(RuntimeError,match="no CPU fallback"):
        OnnxPolicy(None,"cuda:0")
