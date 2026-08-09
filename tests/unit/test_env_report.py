"""
Unit tests for env_report.py.

Tests environment metadata collection without requiring a simulator or WSL2.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from action_retrieval.utils.env_report import (
    EnvironmentInfo,
    collect_environment_info,
    get_git_info,
    get_memory_info,
    get_package_versions,
    get_torch_info,
    write_environment_json,
)


class TestEnvironmentInfo:
    """Tests for EnvironmentInfo dataclass."""

    def test_to_dict(self):
        """EnvironmentInfo.to_dict() returns a valid dict."""
        info = EnvironmentInfo(
            os_name="Linux",
            os_release="5.10.0",
            platform_machine="x86_64",
            python_version="3.10.0",
            python_executable="/usr/bin/python3",
        )
        d = info.to_dict()
        assert isinstance(d, dict)
        assert d["os_name"] == "Linux"
        assert d["python_version"] == "3.10.0"
        assert d["cuda_available"] is False  # default

    def test_to_json(self):
        """EnvironmentInfo.to_json() produces valid JSON."""
        info = EnvironmentInfo(
            os_name="Windows",
            os_release="10.0.26200",
            platform_machine="AMD64",
            python_version="3.12.0",
            python_executable="C:\\Python312\\python.exe",
            torch_version="2.5.1+cpu",
            cuda_available=False,
        )
        json_str = info.to_json()
        parsed = json.loads(json_str)
        assert parsed["os_name"] == "Windows"
        assert parsed["torch_version"] == "2.5.1+cpu"
        assert parsed["cuda_available"] is False


class TestGetTorchInfo:
    """Tests for get_torch_info()."""

    def test_torch_imported(self):
        """If torch is available, return version and CUDA status."""
        # This test only passes if torch is installed (which it should be).
        version, cuda_avail, cuda_ver, cudnn_ver = get_torch_info()
        assert version is not None
        assert isinstance(cuda_avail, bool)

    @patch("builtins.__import__")
    def test_torch_import_error(self, mock_import):
        """If torch import fails, return None values."""
        mock_import.side_effect = ImportError("torch not installed")
        version, cuda_avail, cuda_ver, cudnn_ver = get_torch_info()
        assert version is None
        assert cuda_avail is False
        assert cuda_ver is None
        assert cudnn_ver is None


class TestGetGitInfo:
    """Tests for get_git_info()."""

    def test_git_info_valid_repo(self):
        """In a valid git repo, return commit, branch, and dirty status."""
        # This test runs in the actual repo
        commit, branch, dirty = get_git_info()
        # Commit should be a hex string (if repo has commits)
        if commit:
            assert len(commit) == 40  # SHA-1 hex digest
        # Branch should be a string (at minimum "HEAD" if detached, or "main"/"master")
        if branch:
            assert isinstance(branch, str)
        # Dirty should be a bool
        assert isinstance(dirty, bool)

    def test_git_info_no_git_dir(self):
        """If .git dir is missing, return None values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a temp dir without .git
            commit, branch, dirty = get_git_info(Path(tmpdir))
            assert commit is None
            assert branch is None
            assert dirty is False


class TestGetMemoryInfo:
    """Tests for get_memory_info()."""

    def test_memory_info(self):
        """get_memory_info() returns floats or None."""
        try:
            import psutil  # noqa: F401
            mem_gb, disk_gb = get_memory_info()
            # If psutil is available, should return floats (or None on error)
            if mem_gb is not None:
                assert isinstance(mem_gb, float)
                assert mem_gb > 0
            if disk_gb is not None:
                assert isinstance(disk_gb, float)
                assert disk_gb > 0
        except ImportError:
            # If psutil is not installed, should return None values
            mem_gb, disk_gb = get_memory_info()
            assert mem_gb is None or isinstance(mem_gb, float)
            assert disk_gb is None or isinstance(disk_gb, float)


class TestGetPackageVersions:
    """Tests for get_package_versions()."""

    def test_package_versions(self):
        """get_package_versions() returns a dict of package version strings."""
        versions = get_package_versions()
        assert isinstance(versions, dict)
        # At minimum, we expect numpy and torch to be present
        assert "numpy_version" in versions
        # Values should be strings or None
        for key, val in versions.items():
            assert isinstance(val, (str, type(None)))


class TestCollectEnvironmentInfo:
    """Tests for collect_environment_info()."""

    def test_collect_environment_info(self):
        """collect_environment_info() returns a valid EnvironmentInfo."""
        info = collect_environment_info()
        assert isinstance(info, EnvironmentInfo)
        # Required fields should always be present
        assert info.os_name is not None
        assert info.python_version is not None
        assert isinstance(info.cuda_available, bool)


class TestWriteEnvironmentJson:
    """Tests for write_environment_json()."""

    def test_write_environment_json(self):
        """write_environment_json() creates a valid JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "env.json"
            write_environment_json(output_path)
            assert output_path.exists()
            # Parse JSON
            with open(output_path) as f:
                data = json.load(f)
            assert isinstance(data, dict)
            assert "os_name" in data
            assert "python_version" in data
