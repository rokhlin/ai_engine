import sys
import subprocess
from pathlib import Path
import pytest
import manage

def test_resolve_env_file_default(tmp_path, monkeypatch):
    # Test fallback resolution logic
    monkeypatch.setattr(manage, "PROJECT_ROOT", tmp_path)
    
    # When no file exists
    assert manage.resolve_env_file(None) == tmp_path / "data" / "config" / ".env"
    
    config_dir = tmp_path / "data" / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # When .env exists in data/config
    env_file = config_dir / ".env"
    env_file.write_text("FOO=BAR\n", encoding="utf-8")
    assert manage.resolve_env_file(None) == env_file
    
    # When .env.local exists in data/config, it takes precedence
    env_local = config_dir / ".env.local"
    env_local.write_text("FOO=BAZ\n", encoding="utf-8")
    assert manage.resolve_env_file(None) == env_local

def test_resolve_env_file_custom(tmp_path, monkeypatch):
    monkeypatch.setattr(manage, "PROJECT_ROOT", tmp_path)
    custom = tmp_path / "custom.env"
    custom.write_text("API_PORT=9000\n", encoding="utf-8")
    
    resolved = manage.resolve_env_file(str(custom))
    assert resolved == custom

def test_load_env_vars(tmp_path):
    env_file = tmp_path / "test.env"
    env_file.write_text("""
# Comment line
KEY_ONE=value1
export KEY_TWO="value 2"
KEY_THREE='value 3'
""", encoding="utf-8")
    
    vars_dict = manage.load_env_vars(env_file)
    assert vars_dict["KEY_ONE"] == "value1"
    assert vars_dict["KEY_TWO"] == "value 2"
    assert vars_dict["KEY_THREE"] == "value 3"
    assert "#" not in vars_dict

def test_manage_info_command(capsys):
    env_file = manage.resolve_env_file(None)
    ret = manage.cmd_info(env_file, [])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Media Cataloger Environment Info" in captured.out

def test_command_map_separated_services():
    assert "up-cataloger" in manage.COMMAND_MAP
    assert "up-frontend" in manage.COMMAND_MAP
    assert "up-all" in manage.COMMAND_MAP
    assert "build-cataloger" in manage.COMMAND_MAP
    assert "build-frontend" in manage.COMMAND_MAP
    assert "cataloger" in manage.COMMAND_MAP
    assert "frontend" in manage.COMMAND_MAP
    assert "backend" in manage.COMMAND_MAP

def test_up_commands_dry_run(monkeypatch):
    executed_cmds = []
    def fake_run_cmd(cmd, env_file=None, extra_env=None):
        executed_cmds.append(cmd)
        return 0

    monkeypatch.setattr(manage, "run_cmd", fake_run_cmd)
    env_file = manage.resolve_env_file(None)

    manage.cmd_up_cataloger(env_file, [])
    assert any("docker-compose.yml" in str(c) for c in executed_cmds[-1])

    ret_fe = manage.cmd_up_frontend(env_file, [])
    assert ret_fe == 0

    manage.cmd_up_all(env_file, [])
    assert any("docker-compose.all.yml" in str(c) for c in executed_cmds[-1])

