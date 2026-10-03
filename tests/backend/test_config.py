"""
tests/backend/test_config.py — backend/core/config.py's DATABASE_URL
normalization, the one thing in that module with real logic rather than a
plain os.getenv() passthrough.
"""

import importlib

import backend.core.config as backend_config


def _reload_with(monkeypatch, database_url: str) -> str:
    monkeypatch.setenv("DATABASE_URL", database_url)
    importlib.reload(backend_config)
    return backend_config.DATABASE_URL


class TestDatabaseUrlNormalization:
    """
    Render (and other managed-Postgres hosts following Heroku's old
    convention) hand out "postgres://" connection strings. SQLAlchemy 1.4+
    only accepts "postgresql://" and raises NoSuchModuleError on the old
    scheme — a real, well-documented deploy-time failure if this isn't
    normalized here.
    """

    def test_postgres_scheme_is_rewritten_to_postgresql(self, monkeypatch):
        result = _reload_with(monkeypatch, "postgres://user:pass@host:5432/db")
        assert result == "postgresql://user:pass@host:5432/db"

    def test_postgresql_scheme_is_left_alone(self, monkeypatch):
        result = _reload_with(monkeypatch, "postgresql://user:pass@host:5432/db")
        assert result == "postgresql://user:pass@host:5432/db"

    def test_sqlite_url_is_unaffected(self, monkeypatch):
        result = _reload_with(monkeypatch, "sqlite:///./somewhere.db")
        assert result == "sqlite:///./somewhere.db"

    def test_only_the_leading_scheme_is_replaced(self, monkeypatch):
        """A password or db name that happens to contain the substring
        'postgres://' later in the string must not also get rewritten."""
        result = _reload_with(monkeypatch, "postgres://user:postgres://pw@host/db")
        assert result == "postgresql://user:postgres://pw@host/db"
