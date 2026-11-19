import os
import pytest
from blinkmimic.deploy import graphics

def setup(monkeypatch):
    monkeypatch.setenv("DISPLAY",":test")
    monkeypatch.delenv("__NV_PRIME_RENDER_OFFLOAD",raising=False)
    monkeypatch.delenv("__GLX_VENDOR_LIBRARY_NAME",raising=False)
    monkeypatch.setattr(graphics.shutil,"which",lambda x:"/usr/bin/glxinfo")

def test_auto_replaces_software_only_for_current_process(monkeypatch):
    setup(monkeypatch)
    def probe(env):
        return "NVIDIA GeForce RTX 4090" if env.get("__GLX_VENDOR_LIBRARY_NAME")=="nvidia" else "llvmpipe"
    monkeypatch.setattr(graphics,"probe",probe)
    r=graphics.configure_opengl()
    assert not r["software"] and os.environ["__NV_PRIME_RENDER_OFFLOAD"]=="1"

def test_hardware_default_kept(monkeypatch):
    setup(monkeypatch);monkeypatch.setattr(graphics,"probe",lambda e:"Mesa Intel Iris")
    graphics.configure_opengl()
    assert "__GLX_VENDOR_LIBRARY_NAME" not in os.environ

def test_system_mode_respects_current_renderer(monkeypatch):
    setup(monkeypatch);monkeypatch.setattr(graphics,"probe",lambda e:"llvmpipe")
    assert graphics.configure_opengl("system")["software"]
    assert "__GLX_VENDOR_LIBRARY_NAME" not in os.environ

def test_explicit_nvidia_does_not_silently_fall_back(monkeypatch):
    setup(monkeypatch);monkeypatch.setattr(graphics,"probe",lambda e:"llvmpipe")
    with pytest.raises(RuntimeError,match="not selected"):graphics.configure_opengl("nvidia")
