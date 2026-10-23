import hashlib
import json
from unittest.mock import AsyncMock

import pytest

from scripts.eval_ingestion import clean, load_fixtures


@pytest.mark.parametrize("filename", ["../user.pdf", "subdir/file.docx", "C:\\user\\file.pdf"])
def test_fixture_names_cannot_escape_the_declared_directory(tmp_path, filename):
    (tmp_path / "manifest.json").write_text(json.dumps({"files": [{"name": filename}]}))
    with pytest.raises(ValueError, match="filename"):
        load_fixtures(tmp_path)


def test_changed_fixture_is_rejected_before_model_calls_or_upload(tmp_path):
    (tmp_path / "test.pdf").write_bytes(b"changed file")
    (tmp_path / "manifest.json").write_text(json.dumps({"files": [{"name": "test.pdf", "sha256": hashlib.sha256(b"original").hexdigest()}]}))
    with pytest.raises(ValueError, match="hash_mismatch"):
        load_fixtures(tmp_path)


async def test_cleanup_never_targets_an_existing_user_even_if_called_directly():
    client, store = AsyncMock(), AsyncMock()
    with pytest.raises(ValueError, match="namespace"):
        await clean(client, store, "admin", ["user.pdf"], ["user-session"])
    client.delete.assert_not_called()
    store._run.assert_not_called()
