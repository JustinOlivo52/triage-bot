"""
tests/backend/test_observability.py — backend/core/observability.py's
structured log formatting and the SENTRY_DSN-gated error tracking init.

No real Sentry project needed: init_error_tracking() calling sentry_sdk.init
is monkeypatched out, same reason test_config.py never touches a real
Postgres — the point under test is "did we call it with the right thing",
not the SDK's own behavior.
"""

import importlib
import json
import logging

import backend.core.config as backend_config
import backend.core.observability as observability


def _reload_with(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    importlib.reload(backend_config)
    importlib.reload(observability)


class TestJsonFormatter:
    def test_formats_record_as_valid_json_with_expected_fields(self):
        formatter = observability.JsonFormatter()
        record = logging.LogRecord(
            name="backend.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="patient checked in",
            args=(),
            exc_info=None,
        )
        parsed = json.loads(formatter.format(record))
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "backend.test"
        assert parsed["message"] == "patient checked in"
        assert "timestamp" in parsed
        assert "exc_info" not in parsed

    def test_includes_exc_info_when_present(self):
        formatter = observability.JsonFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="backend.test",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="triage call failed",
                args=(),
                exc_info=sys.exc_info(),
            )
        parsed = json.loads(formatter.format(record))
        assert "ValueError: boom" in parsed["exc_info"]


class TestConfigureLogging:
    def test_sets_root_level_from_log_level_env(self, monkeypatch):
        _reload_with(monkeypatch, LOG_LEVEL="WARNING")
        observability.configure_logging()
        assert logging.getLogger().level == logging.WARNING

    def test_installs_exactly_one_json_handler(self, monkeypatch):
        _reload_with(monkeypatch, LOG_LEVEL="INFO")
        observability.configure_logging()
        observability.configure_logging()  # idempotent — must not stack handlers
        handlers = logging.getLogger().handlers
        assert len(handlers) == 1
        assert isinstance(handlers[0].formatter, observability.JsonFormatter)


class TestInitErrorTracking:
    def test_noop_when_sentry_dsn_unset(self, monkeypatch):
        _reload_with(monkeypatch, SENTRY_DSN="")
        import sentry_sdk

        calls = []
        monkeypatch.setattr(sentry_sdk, "init", lambda **kwargs: calls.append(kwargs))

        observability.init_error_tracking()

        assert calls == []

    def test_calls_sentry_init_with_dsn_when_set(self, monkeypatch):
        _reload_with(monkeypatch, SENTRY_DSN="https://key@o0.ingest.sentry.io/0", DEMO_MODE="true")
        import sentry_sdk

        calls = []
        monkeypatch.setattr(sentry_sdk, "init", lambda **kwargs: calls.append(kwargs))

        observability.init_error_tracking()

        assert len(calls) == 1
        assert calls[0]["dsn"] == "https://key@o0.ingest.sentry.io/0"
        assert calls[0]["environment"] == "demo"
        assert calls[0]["traces_sample_rate"] == 0.0

    def test_environment_is_production_when_demo_mode_off(self, monkeypatch):
        _reload_with(monkeypatch, SENTRY_DSN="https://key@o0.ingest.sentry.io/0", DEMO_MODE="false")
        import sentry_sdk

        calls = []
        monkeypatch.setattr(sentry_sdk, "init", lambda **kwargs: calls.append(kwargs))

        observability.init_error_tracking()

        assert calls[0]["environment"] == "production"
