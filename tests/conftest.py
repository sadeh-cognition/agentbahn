from __future__ import annotations

import os
import sys
from pathlib import Path


os.environ.setdefault("LLM_API_KEY_ENCRYPTION_KEY", "test-llm-api-key-encryption-key")
os.environ.setdefault("CODEBASE_AGENT_COST_LIMIT", "0.25")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
