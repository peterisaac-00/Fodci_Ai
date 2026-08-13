from __future__ import annotations

import backend_ai


def test_package_exposes_a_version_and_settings_loader() -> None:
    assert backend_ai.__version__ == "0.1.0"
    assert callable(backend_ai.load_settings)
