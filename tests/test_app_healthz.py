import os
import unittest
from unittest.mock import patch

import app as app_module


class HealthzTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()

    def test_healthz_returns_ok_when_dependencies_are_configured(self) -> None:
        with patch.object(app_module, "DRIVE_REVIEW_QUEUE_ROOT_ID", "review-root"), patch.object(
            app_module,
            "DRIVE_PROJECT_ROOT_FOLDER_ID",
            None,
        ), patch.object(
            app_module,
            "db_healthcheck",
            return_value={"enabled": True, "healthy": True, "error": None},
        ), patch.dict(os.environ, {"DRIVE_SERVICE_ACCOUNT_JSON": "{}"}, clear=False):
            response = self.client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertTrue(payload["drive_configured"])
        self.assertTrue(payload["review_root_configured"])

    def test_healthz_returns_503_when_database_is_unhealthy(self) -> None:
        with patch.object(app_module, "DRIVE_REVIEW_QUEUE_ROOT_ID", "review-root"), patch.object(
            app_module,
            "DRIVE_PROJECT_ROOT_FOLDER_ID",
            None,
        ), patch.object(
            app_module,
            "db_healthcheck",
            return_value={"enabled": True, "healthy": False, "error": "db down"},
        ), patch.dict(os.environ, {"DRIVE_SERVICE_ACCOUNT_JSON": "{}"}, clear=False):
            response = self.client.get("/healthz")

        self.assertEqual(response.status_code, 503)
        payload = response.get_json()
        self.assertFalse(payload["success"])
        self.assertEqual(payload["database"]["error"], "db down")
