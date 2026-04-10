import pytest

import supabase_client
from supabase_client import Environment, SupabaseConfig, get_connection_status, with_supabase_client


def test_supabase_config_uses_environment_fallbacks(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://prod.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "prod-key")
    monkeypatch.delenv("DEV_SUPABASE_URL", raising=False)
    monkeypatch.delenv("DEV_SUPABASE_KEY", raising=False)

    config = SupabaseConfig.get_config(Environment.DEVELOPMENT)

    assert config["url"] == "https://prod.supabase.co"
    assert config["key"] == "prod-key"
    assert config["max_retries"] == 5


def test_supabase_config_raises_when_credentials_are_missing(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)

    with pytest.raises(ValueError):
        SupabaseConfig.get_config(Environment.PRODUCTION)


def test_with_supabase_client_injects_manager_client(monkeypatch):
    class DummyManager:
        def __init__(self, _env):
            self.client = "stub-client"

    monkeypatch.setattr(supabase_client, "SupabaseManager", DummyManager)

    @with_supabase_client(Environment.PRODUCTION)
    def sample_function(client, value):
        return client, value

    assert sample_function(42) == ("stub-client", 42)


def test_get_connection_status_returns_connected_metrics(monkeypatch):
    class DummyPool:
        def is_healthy(self):
            return True

    class DummyManager:
        def __init__(self, _env):
            self.connection_pool = DummyPool()

        def get_metrics(self):
            return {"total_queries": 3, "failed_queries": 0}

    monkeypatch.setattr(supabase_client, "SupabaseManager", DummyManager)

    status = get_connection_status()

    assert status == {
        "connected": True,
        "metrics": {"total_queries": 3, "failed_queries": 0},
    }
