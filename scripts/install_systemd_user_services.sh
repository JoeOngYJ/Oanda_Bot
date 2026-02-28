#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${1:-/home/joe/Desktop/Algo_trading/oanda-trading-system}"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
TEMPLATE_DIR="$PROJECT_ROOT/deploy/systemd"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

mkdir -p "$UNIT_DIR"

install_unit() {
  local name="$1"
  local src="$TEMPLATE_DIR/$name.tmpl"
  local dst="$UNIT_DIR/$name"
  if [ ! -f "$src" ]; then
    echo "Missing template: $src" >&2
    exit 1
  fi
  sed "s#__PROJECT_ROOT__#$PROJECT_ROOT#g" "$src" > "$dst"
  echo "Installed $dst"
}

install_unit "oanda-infra.service"
install_unit "oanda-trading-supervisor.service"
install_unit "oanda-discord-operator.service"

systemctl --user daemon-reload

cat <<EOF

Installed user services in: $UNIT_DIR

Next commands:
  systemctl --user enable --now oanda-trading-supervisor.service
  systemctl --user enable --now oanda-discord-operator.service
  # optional if you want docker-compose infra managed by systemd
  systemctl --user enable --now oanda-infra.service
  systemctl --user status oanda-trading-supervisor.service oanda-discord-operator.service

Recommended for laptop reboot persistence:
  sudo loginctl enable-linger $USER

EOF
