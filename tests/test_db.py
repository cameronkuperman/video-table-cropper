import os
import unittest
from unittest.mock import patch

import db


class DatabaseHelperTests(unittest.TestCase):
    def test_database_disabled_without_database_url(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(db.database_enabled())
            self.assertEqual(
                db.db_healthcheck(),
                {"enabled": False, "healthy": True, "error": None},
            )

    def test_database_url_is_trimmed(self) -> None:
        with patch.dict(os.environ, {"DATABASE_URL": "  postgres://example  "}, clear=True):
            self.assertEqual(db.get_database_url(), "postgres://example")
            self.assertTrue(db.database_enabled())
