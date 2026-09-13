"""Remote SSH execution helper using Paramiko.

Connects to the remote Blackwell box using credentials stored in a local
gitignored config file (or environment variables) and executes shell commands,
streaming output in real time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import paramiko


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CONFIG_FILE = Path(__file__).resolve().parents[2] / ".remote_box.json"


def load_config() -> dict[str, str | int]:
    """Load remote connection config from .remote_box.json or environment."""
    cfg = {}
    if CONFIG_FILE.exists():
        try:
            cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Failed to read {CONFIG_FILE}: {e}", file=sys.stderr)

    host = os.environ.get("REMOTE_HOST", cfg.get("host", ""))
    port = int(os.environ.get("REMOTE_PORT", cfg.get("port", 22)))
    user = os.environ.get("REMOTE_USER", cfg.get("user", "root"))
    password = os.environ.get("REMOTE_PASSWORD", cfg.get("password", ""))

    return {"host": host, "port": port, "user": user, "password": password}


def save_config(host: str, port: int, user: str, password: str) -> None:
    """Save connection config to .remote_box.json (gitignored at root)."""
    cfg = {"host": host, "port": port, "user": user, "password": password}
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Saved connection configuration to {CONFIG_FILE}")


def get_client(cfg: dict[str, str | int]) -> paramiko.SSHClient:
    """Create and connect an SSHClient."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=str(cfg["host"]),
        port=int(cfg["port"]),
        username=str(cfg["user"]),
        password=str(cfg["password"]) if cfg["password"] else None,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )
    return client


def run_remote_command(cmd: str, cfg: dict[str, str | int] | None = None) -> int:
    """Execute a command on the remote host and stream stdout/stderr."""
    if cfg is None:
        cfg = load_config()

    if not cfg.get("host"):
        print("[ERROR] Remote host not configured. Run with --configure first.", file=sys.stderr)
        return 1

    client = get_client(cfg)
    try:
        stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
        for line in iter(stdout.readline, ""):
            print(line, end="", flush=True)

        exit_status = stdout.channel.recv_exit_status()
        return exit_status
    finally:
        client.close()


def setup_authorized_keys(cfg: dict[str, str | int]) -> None:
    """Optionally inject local SSH public key into remote ~/.ssh/authorized_keys."""
    pub_key_path = Path.home() / ".ssh" / "id_ed25519.pub"
    if not pub_key_path.exists():
        pub_key_path = Path.home() / ".ssh" / "id_rsa.pub"

    if not pub_key_path.exists():
        print(f"[INFO] No local public key found at {pub_key_path}. Skipping key injection.")
        return

    pub_key = pub_key_path.read_text(encoding="utf-8").strip()
    client = get_client(cfg)
    try:
        setup_cmd = (
            f"mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
            f'grep -qxF "{pub_key}" ~/.ssh/authorized_keys 2>/dev/null || '
            f'echo "{pub_key}" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
        )
        _, stdout, _ = client.exec_command(setup_cmd)
        stdout.channel.recv_exit_status()
        print("[INFO] Local SSH public key successfully added to remote authorized_keys.")
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote SSH runner for Blackwell box")
    parser.add_argument("--configure", action="store_true", help="Configure connection parameters")
    parser.add_argument("--host", type=str, help="Remote host/IP")
    parser.add_argument("--port", type=int, default=22, help="Remote SSH port (default: 22)")
    parser.add_argument("--user", type=str, default="root", help="Remote username (default: root)")
    parser.add_argument("--password", type=str, help="Remote password")
    parser.add_argument("--setup-keys", action="store_true", help="Inject local public key into remote authorized_keys")
    parser.add_argument("command", nargs="*", help="Command to execute on the remote box")

    args = parser.parse_args()

    if args.configure or (args.host and args.password):
        if not args.host or not args.password:
            print("[ERROR] --host and --password are required for configuration", file=sys.stderr)
            sys.exit(1)
        save_config(args.host, args.port, args.user, args.password)
        if args.setup_keys:
            setup_authorized_keys(load_config())
        sys.exit(0)

    cfg = load_config()
    if not cfg.get("host"):
        print("[ERROR] No remote host configured. Run with --configure --host <IP> --password <PW>", file=sys.stderr)
        sys.exit(1)

    if args.setup_keys:
        setup_authorized_keys(cfg)
        sys.exit(0)

    if not args.command:
        print("[ERROR] No command specified to run.", file=sys.stderr)
        sys.exit(1)

    full_cmd = " ".join(args.command)
    exit_code = run_remote_command(full_cmd, cfg)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
