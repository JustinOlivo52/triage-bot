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
    scheme. Beyond that, the driver also has to be named explicitly
    ("+psycopg2") rather than left for SQLAlchemy to resolve by default —
    a real deploy failure showed that default isn't stable across
    SQLAlchemy versions: 2.1 picked the psycopg (v3) dialect for a bare
    "postgresql://" URL, which isn't installed (only psycopg2-binary is),
    and crashed `alembic upgrade head` with ModuleNotFoundError.
    """

    def test_postgres_scheme_is_rewritten_to_postgresql_psycopg2(self, monkeypatch):
        result = _reload_with(monkeypatch, "postgres://user:pass@host:5432/db")
        assert result == "postgresql+psycopg2://user:pass@host:5432/db"

    def test_bare_postgresql_scheme_also_gets_the_driver_named(self, monkeypatch):
        result = _reload_with(monkeypatch, "postgresql://user:pass@host:5432/db")
        assert result == "postgresql+psycopg2://user:pass@host:5432/db"

    def test_postgresql_psycopg2_scheme_is_left_alone(self, monkeypatch):
        result = _reload_with(monkeypatch, "postgresql+psycopg2://user:pass@host:5432/db")
        assert result == "postgresql+psycopg2://user:pass@host:5432/db"

    def test_sqlite_url_is_unaffected(self, monkeypatch):
        result = _reload_with(monkeypatch, "sqlite:///./somewhere.db")
        assert result == "sqlite:///./somewhere.db"

    def test_only_the_leading_scheme_is_replaced(self, monkeypatch):
        """A password or db name that happens to contain the substring
        'postgres://' later in the string must not also get rewritten."""
        result = _reload_with(monkeypatch, "postgres://user:postgres://pw@host/db")
        assert result == "postgresql+psycopg2://user:postgres://pw@host/db"
