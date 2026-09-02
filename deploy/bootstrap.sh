#!/usr/bin/env bash
#
# One-shot VPS setup for a forward walk. Idempotent: safe to re-run.
#
#   sudo ./deploy/bootstrap.sh
#
# Creates a service user, clones the repo, builds the venv, installs the
# systemd units. It does NOT start anything and it does NOT install IB
# Gateway -- run deploy/install-ibc.sh for that, then `deltahedger doctor`
# before you let it trade.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/vivere7108-lab/DeltaHedgedShortVol.git}"
BRANCH="${BRANCH:-main}"
SERVICE_USER="${SERVICE_USER:-deltahedger}"
INSTALL_DIR="${INSTALL_DIR:-/opt/deltahedger}"

log() { printf '\n== %s\n' "$*"; }

if [[ $EUID -ne 0 ]]; then
    echo "run as root: sudo $0" >&2
    exit 1
fi

log "system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# xvfb/openbox are for IB Gateway: it is a Java GUI app with no headless mode,
# so it needs a virtual display even when nobody is looking at it.
apt-get install -yqq \
    python3 python3-venv python3-dev build-essential \
    git curl unzip \
    xvfb x11vnc openbox \
    tzdata

log "service user: ${SERVICE_USER}"
if ! id -u "${SERVICE_USER}" >/dev/null 2>&1; then
    useradd --system --create-home --home-dir "/home/${SERVICE_USER}" \
            --shell /usr/sbin/nologin "${SERVICE_USER}"
else
    echo "already exists"
fi

log "repository: ${INSTALL_DIR} (${BRANCH})"

# Git refuses to operate on a repository owned by someone else ("detected
# dubious ownership"), which is exactly what this script creates: it clones,
# then chowns to the service user. Running the update as root would then
# fail on every re-run -- so run git AS the owner, and pass safe.directory
# inline (not into anyone's global config) to survive a tree whose ownership
# is already mixed from an earlier root pull.
as_owner() {
    sudo -u "${SERVICE_USER}" env HOME="/home/${SERVICE_USER}" \
        git -c safe.directory="${INSTALL_DIR}" "$@"
}

mkdir -p "${INSTALL_DIR}"
# Normalise ownership before touching git, so the service user can act on
# whatever a previous run (or a manual root pull) left behind.
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${INSTALL_DIR}"

if [[ -d "${INSTALL_DIR}/.git" ]]; then
    as_owner -C "${INSTALL_DIR}" fetch --quiet origin "${BRANCH}"
    as_owner -C "${INSTALL_DIR}" checkout --quiet "${BRANCH}"
    as_owner -C "${INSTALL_DIR}" reset --hard --quiet "origin/${BRANCH}"
else
    as_owner clone --quiet --branch "${BRANCH}" "${REPO_URL}" "${INSTALL_DIR}"
fi
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${INSTALL_DIR}"

# Say what actually landed. "Did the patch reach the box?" is otherwise only
# answerable by reading the script, and a silently-stale checkout looks
# identical to a fixed one until it fails the same way twice.
echo "  now at $(as_owner -C "${INSTALL_DIR}" log -1 --format='%h %s')"

log "python environment"
sudo -u "${SERVICE_USER}" python3 -m venv "${INSTALL_DIR}/.venv"
sudo -u "${SERVICE_USER}" "${INSTALL_DIR}/.venv/bin/pip" install --quiet --upgrade pip
sudo -u "${SERVICE_USER}" "${INSTALL_DIR}/.venv/bin/pip" install --quiet -e "${INSTALL_DIR}[ibkr]"

log "run directory"
install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 750 "${INSTALL_DIR}/runs"

log "systemd units"
for unit in ibc.service deltahedger.service; do
    sed -e "s|@INSTALL_DIR@|${INSTALL_DIR}|g" \
        -e "s|@SERVICE_USER@|${SERVICE_USER}|g" \
        "${INSTALL_DIR}/deploy/${unit}" > "/etc/systemd/system/${unit}"
done
if [[ ! -f /etc/deltahedger.env ]]; then
    install -o root -g "${SERVICE_USER}" -m 640 \
        "${INSTALL_DIR}/deploy/deltahedger.env.example" /etc/deltahedger.env
    echo "wrote /etc/deltahedger.env -- review it before starting"
else
    echo "/etc/deltahedger.env exists; left alone"
fi
systemctl daemon-reload

cat <<EOF

Bootstrap done. Nothing is running yet, by design.

Next:
  1. sudo ${INSTALL_DIR}/deploy/install-ibc.sh      # IB Gateway + IBC
  2. sudo nano /etc/ibc/config.ini                  # paper credentials, mode=paper
  3. sudo systemctl enable --now ibc                # start the gateway
  4. sudo -u ${SERVICE_USER} ${INSTALL_DIR}/.venv/bin/deltahedger doctor \\
         -c ${INSTALL_DIR}/configs/es_paper.yaml    # must pass before trading
  5. sudo systemctl enable --now deltahedger        # starts in --dry-run

Read ${INSTALL_DIR}/deploy/README.md before step 5.
EOF
