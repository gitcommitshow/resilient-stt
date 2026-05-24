"""Make repo root importable so `import core` etc. works in tests without install."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.privacy import disable_dependency_telemetry  # noqa: E402

disable_dependency_telemetry()
