#!/usr/bin/env python
"""
ISTR Installation Script

This setup.py handles the installation of:
1. Detectron2 (from the detectron2/ subdirectory as a local dependency - MUST be editable)
2. ISTR (as a separate package that uses detectron2)
3. ISTR configs are injected into detectron2.model_zoo for seamless integration

Installation:
    pip install -e . --no-build-isolation

The ISTR configs will be available through detectron2's model zoo:
    from detectron2 import model_zoo
    cfg.merge_from_file(model_zoo.get_config_file("ISTR/ISTR-AE-R50-3x.yaml"))
"""

import glob
import os
import shutil
from pathlib import Path
from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.build_py import build_py


def get_istr_version():
    """Get ISTR version."""
    version_file = Path(__file__).parent / "projects" / "ISTR" / "istr" / "__init__.py"
    if version_file.exists():
        with open(version_file, "r") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[-1].strip().strip("'\"")
    return "1.0.0"


def inject_istr_configs_to_detectron2():
    """
    Copy ISTR configs into detectron2's model_zoo configs directory.
    This allows ISTR configs to be accessible via detectron2.model_zoo.
    """
    # Source: ISTR configs
    istr_configs_src = Path(__file__).parent / "projects" / "ISTR" / "configs"
    
    # Destination: detectron2's model_zoo configs
    detectron2_configs = Path(__file__).parent / "detectron2" / "detectron2" / "model_zoo" / "configs"
    istr_configs_dest = detectron2_configs / "ISTR"
    
    if not istr_configs_src.exists():
        print(f"Warning: ISTR configs not found at {istr_configs_src}")
        return
    
    if not detectron2_configs.exists():
        print(f"Warning: Detectron2 model_zoo configs not found at {detectron2_configs}")
        print("Make sure detectron2 is installed in editable mode first!")
        return
    
    # Remove old ISTR configs if they exist
    if istr_configs_dest.exists():
        if istr_configs_dest.is_symlink():
            istr_configs_dest.unlink()
        elif istr_configs_dest.is_dir():
            shutil.rmtree(istr_configs_dest)
    
    # Create symlink or copy configs
    try:
        # Try to create a symlink (preferred for editable installs)
        istr_configs_dest.symlink_to(istr_configs_src.absolute(), target_is_directory=True)
        print(f"✓ Symlinked ISTR configs to detectron2.model_zoo: {istr_configs_dest}")
    except (OSError, NotImplementedError):
        # Fall back to copying if symlink fails (e.g., on Windows)
        shutil.copytree(istr_configs_src, istr_configs_dest)
        print(f"✓ Copied ISTR configs to detectron2.model_zoo: {istr_configs_dest}")


class CustomDevelop(develop):
    """Custom develop command that injects ISTR configs into detectron2."""
    
    def run(self):
        develop.run(self)
        inject_istr_configs_to_detectron2()


class CustomBuildPy(build_py):
    """Custom build command that injects ISTR configs into detectron2."""
    
    def run(self):
        build_py.run(self)
        inject_istr_configs_to_detectron2()


# Find ISTR package - the istr package is directly at projects/ISTR/istr
# We need to tell setuptools where to find it
istr_packages = find_packages(where="projects/ISTR", exclude=("test*", "demo*"))

# Get absolute path to detectron2 subdirectory for installation
detectron2_path = Path(__file__).parent.absolute() / "detectron2"

# Use PEP 621 metadata from pyproject.toml. Keep only custom commands here.
setup(
    # setuptools will read metadata (name, version, dependencies, packages, etc.)
    # from `pyproject.toml` (PEP 621). We keep custom build/develop commands here
    # so they run during `pip install -e .` or builds.
    cmdclass={
        "develop": CustomDevelop,
        "build_py": CustomBuildPy,
    },
)