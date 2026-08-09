"""
Environment and reproducibility metadata collection.

Collects OS/Python/torch/CUDA versions, git state, available RAM/disk, and
package versions — used by smoke tests and experiment logging per spec 17.4.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


class EnvironmentInfo:
    """Container for reproducibility metadata."""
    def __init__(self, os_name: str, os_release: str, platform_machine: str,
                 python_version: str, python_executable: str, **kwargs):
        self.os_name = os_name
        self.os_release = os_release
        self.platform_machine = platform_machine
        self.python_version = python_version
        self.python_executable = python_executable

        self.torch_version = kwargs.get('torch_version')
        self.cuda_available = kwargs.get('cuda_available', False)
        self.cuda_version = kwargs.get('cuda_version')
        self.cudnn_version = kwargs.get('cudnn_version')

        self.git_commit = kwargs.get('git_commit')
        self.git_branch = kwargs.get('git_branch')
        self.git_dirty = kwargs.get('git_dirty', False)

        self.memory_gb_available = kwargs.get('memory_gb_available')
        self.disk_gb_free = kwargs.get('disk_gb_free')

        # Package versions (dynamic)
        self._extra_fields = {k: v for k, v in kwargs.items()
                             if k not in ['torch_version', 'cuda_available', 'cuda_version',
                                         'cudnn_version', 'git_commit', 'git_branch', 'git_dirty',
                                         'memory_gb_available', 'disk_gb_free']}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = {
            'os_name': self.os_name,
            'os_release': self.os_release,
            'platform_machine': self.platform_machine,
            'python_version': self.python_version,
            'python_executable': self.python_executable,
            'torch_version': self.torch_version,
            'cuda_available': self.cuda_available,
            'cuda_version': self.cuda_version,
            'cudnn_version': self.cudnn_version,
            'git_commit': self.git_commit,
            'git_branch': self.git_branch,
            'git_dirty': self.git_dirty,
            'memory_gb_available': self.memory_gb_available,
            'disk_gb_free': self.disk_gb_free,
        }
        d.update(self._extra_fields)
        return d

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def get_torch_info() -> tuple[Optional[str], bool, Optional[str], Optional[str]]:
    """Get torch version, CUDA availability, CUDA version, cuDNN version."""
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else None
        cudnn_version = torch.backends.cudnn.version() if cuda_available else None
        return torch_version, cuda_available, cuda_version, cudnn_version
    except ImportError:
        return None, False, None, None


def get_git_info(repo_root: Optional[Path] = None) -> tuple[Optional[str], Optional[str], bool]:
    """Get git commit hash, branch name, and dirty status."""
    try:
        if repo_root is None:
            repo_root = Path(__file__).parent.parent.parent.parent  # project root

        if not (repo_root / ".git").exists():
            return None, None, False

        os.chdir(repo_root)

        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()

        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()

        # Check if there are uncommitted changes
        dirty = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD"],
            stderr=subprocess.DEVNULL
        ).returncode != 0

        return commit, branch, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None, False


def get_memory_info() -> tuple[Optional[float], Optional[float]]:
    """Get available RAM (GB) and free disk space (GB) in repo root."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        memory_available_gb = mem.available / (1024 ** 3)

        repo_root = Path(__file__).parent.parent.parent.parent
        disk = shutil.disk_usage(repo_root)
        disk_free_gb = disk.free / (1024 ** 3)

        return memory_available_gb, disk_free_gb
    except (ImportError, Exception):
        return None, None


def get_package_versions() -> Dict[str, Optional[str]]:
    """Get versions of key scientific packages."""
    packages = {}

    for pkg_name in ["numpy", "scipy", "pandas", "hydra", "omegaconf", "matplotlib", "Pillow"]:
        try:
            mod = __import__(pkg_name)
            version = getattr(mod, "__version__", None)
            packages[f"{pkg_name}_version"] = version
        except ImportError:
            packages[f"{pkg_name}_version"] = None

    return packages


def collect_environment_info(repo_root: Optional[Path] = None) -> EnvironmentInfo:
    """Collect all reproducibility metadata."""
    torch_version, cuda_available, cuda_version, cudnn_version = get_torch_info()
    git_commit, git_branch, git_dirty = get_git_info(repo_root)
    memory_available_gb, disk_free_gb = get_memory_info()
    pkg_versions = get_package_versions()

    info = EnvironmentInfo(
        os_name=platform.system(),
        os_release=platform.release(),
        platform_machine=platform.machine(),
        python_version=platform.python_version(),
        python_executable=sys.executable,
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
        git_commit=git_commit,
        git_branch=git_branch,
        git_dirty=git_dirty,
        memory_gb_available=memory_available_gb,
        disk_gb_free=disk_free_gb,
        **pkg_versions
    )

    return info


def write_environment_json(output_path: Path) -> None:
    """Collect environment info and write to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info = collect_environment_info()
    with open(output_path, "w") as f:
        f.write(info.to_json())

    print(f"Environment info written to {output_path}")
