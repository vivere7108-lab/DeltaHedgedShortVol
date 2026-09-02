#!/usr/bin/env bash
#
# IB Gateway + IBC, headless.
#
#   sudo ./deploy/install-ibc.sh
#
# Why IBC: IB Gateway is a desktop application that expects a human to type
# a password and to click through the restart it forces on itself every day.
# IBC drives both. Without it an unattended walk dies at the first daily
# restart, and there is nothing in the strategy's log to say why.
#
# This script downloads from Interactive Brokers and from the IBC project.
# Versions move; if a download 404s, check the two URLs below rather than
# assuming the script is broken.

set -euo pipefail

SERVICE_USER="${SERVICE_USER:-deltahedger}"
IBC_VERSION="${IBC_VERSION:-3.20.0}"
IBC_DIR="${IBC_DIR:-/opt/ibc}"
IBC_CONFIG_DIR="${IBC_CONFIG_DIR:-/etc/ibc}"
TWS_SETTINGS_DIR="${TWS_SETTINGS_DIR:-/home/${SERVICE_USER}/Jts}"

GATEWAY_URL="${GATEWAY_URL:-https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh}"
IBC_URL="${IBC_URL:-https://github.com/IbcAlpha/IBC/releases/download/${IBC_VERSION}/IBCLinux-${IBC_VERSION}.zip}"

log() { printf '\n== %s\n' "$*"; }

# Make a path executable BY THE SERVICE USER, not merely by root.
#
# Three different things produce the identical "sudo: unable to execute ...
# Permission denied", and it is worth fixing all of them rather than
# guessing which one bit:
#
#   1. `mktemp -d` as root creates the directory 0700 root-owned, so the
#      service user cannot traverse into it -- the file's own mode is then
#      irrelevant;
#   2. unzip under umask 022 leaves scripts 0644, and `chmod u+x` makes that
#      0744, which is executable by root and by nobody else;
#   3. /tmp is mounted noexec on plenty of hardened VPS images.
#
# Staging under the service user's own home dodges (1) and (3); make_runnable
# handles (2).
make_runnable() {
    local path="$1"
    chown "${SERVICE_USER}:${SERVICE_USER}" "${path}"
    chmod 0755 "${path}"
}

# Fail here, with an explanation, rather than three steps later inside
# systemd where the error has no context.
assert_runnable() {
    local path="$1" what="$2"
    if ! sudo -u "${SERVICE_USER}" test -x "${path}"; then
        cat >&2 <<EOF

${what} is not executable by ${SERVICE_USER}:

  $(stat -c '%a %U:%G %n' "${path}")

Every directory on that path must be traversable by ${SERVICE_USER}, the
file itself must be mode 0755, and the filesystem must not be mounted
noexec. Check:  mount | grep ' $(df --output=target "${path}" | tail -1) '
EOF
        return 1
    fi
}

if [[ $EUID -ne 0 ]]; then
    echo "run as root: sudo $0" >&2
    exit 1
fi

# Which version of this script is actually running. A stale checkout fails
# in exactly the same way as an unfixed one, so say the commit up front
# rather than leaving "did the pull land?" to be inferred from the error.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if commit=$(git -c safe.directory='*' -C "${script_dir}" \
                log -1 --format='%h %ad %s' --date=short 2>/dev/null); then
    echo "install-ibc.sh from ${commit}"
fi

if ! id -u "${SERVICE_USER}" >/dev/null 2>&1; then
    echo "no such user: ${SERVICE_USER}. Run deploy/bootstrap.sh first." >&2
    exit 1
fi

# Staging lives in the service user's home, not /tmp: see make_runnable.
staging="/home/${SERVICE_USER}/.install"
install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 0755 "${staging}"
workdir="$(mktemp -d "${staging}/ibc-XXXXXX")"
chown "${SERVICE_USER}:${SERVICE_USER}" "${workdir}"
chmod 0755 "${workdir}"
trap 'rm -rf "${workdir}"' EXIT

log "IB Gateway"

# IBC locates the gateway as $TWS_PATH/ibgateway/$TWS_MAJOR_VRSN/jars, so the
# install has to land in that exact shape. Passing -dir put the jars flat in
# Jts/ibgateway/ instead, which IBC cannot find -- it fails to launch, nothing
# ever listens on the API port, and the only symptom is a connection refused
# from a component three steps downstream. So: let the installer use its own
# default location, then DETECT what it produced rather than assuming.
GATEWAY_ROOT="/home/${SERVICE_USER}/Jts/ibgateway"

detect_version() {
    # Standard layout -> print the version directory name (e.g. "1030").
    local jars
    jars="$(find "${GATEWAY_ROOT}" -maxdepth 2 -mindepth 2 -type d -name jars \
            2>/dev/null | sort -V | tail -1)"
    [[ -n "${jars}" ]] || return 1
    basename "$(dirname "${jars}")"
}

# A flat install is what the earlier version of this script produced. It is
# unusable by IBC and there is no reliable way to read the version back out
# of it, so clear it and let the installer lay itself out properly.
if [[ -d "${GATEWAY_ROOT}/jars" ]]; then
    log "removing an unusable flat gateway install at ${GATEWAY_ROOT}"
    echo "  (IBC needs ibgateway/<version>/jars; this has ibgateway/jars)"
    mv "${GATEWAY_ROOT}" "${GATEWAY_ROOT}.flat.$(date +%s)"
fi

if TWS_MAJOR_VRSN="$(detect_version)"; then
    echo "already installed: version ${TWS_MAJOR_VRSN}"
else
    curl -fsSL "${GATEWAY_URL}" -o "${workdir}/ibgateway.sh"
    make_runnable "${workdir}/ibgateway.sh"
    assert_runnable "${workdir}/ibgateway.sh" "The IB Gateway installer"

    # HOME and TMPDIR are set explicitly. The installer is an install4j
    # bundle that unpacks itself into TMPDIR before running -- pointing that
    # at the staging directory keeps it off a possibly-noexec /tmp -- and
    # sudo does not reliably hand the target user its own HOME, which is
    # where the installer puts ~/Jts.
    #
    # -q is unattended mode. No -dir: the default is the layout IBC expects.
    sudo -u "${SERVICE_USER}" env \
        HOME="/home/${SERVICE_USER}" \
        TMPDIR="${workdir}" \
        "${workdir}/ibgateway.sh" -q

    if ! TWS_MAJOR_VRSN="$(detect_version)"; then
        cat >&2 <<EOF

The IB Gateway installer ran but no ibgateway/<version>/jars directory
appeared under ${GATEWAY_ROOT}. IBC cannot start without it.

What is actually there:
$(ls -la "${GATEWAY_ROOT}" 2>/dev/null || echo "  (nothing -- the install did not write anything)")
EOF
        exit 1
    fi
    echo "installed version ${TWS_MAJOR_VRSN}"
fi

log "IBC ${IBC_VERSION}"
if [[ -f "${IBC_DIR}/gatewaystart.sh" ]]; then
    echo "already installed; skipping download"
else
    curl -fsSL "${IBC_URL}" -o "${workdir}/ibc.zip"
    mkdir -p "${IBC_DIR}"
    unzip -oq "${workdir}/ibc.zip" -d "${IBC_DIR}"
    # a+rX, not u+x: unzip leaves 0644 and `chmod u+x` would make that 0744,
    # which root can run and the service user cannot -- and ibc.service runs
    # as the service user, so it would fail at ExecStart with no clue why.
    # The capital X sets the directory traverse bit without marking every
    # data file executable.
    chmod -R a+rX "${IBC_DIR}"
    find "${IBC_DIR}" -name '*.sh' -exec chmod 0755 {} +
fi

assert_runnable "${IBC_DIR}/gatewaystart.sh" "The IBC launcher"

log "configuration"
mkdir -p "${IBC_CONFIG_DIR}"
if [[ -f "${IBC_CONFIG_DIR}/config.ini" ]]; then
    echo "${IBC_CONFIG_DIR}/config.ini exists; left alone"
else
    cp "${IBC_DIR}/config.ini" "${IBC_CONFIG_DIR}/config.ini"
    # Sane defaults for an unattended paper walk. Credentials are NOT set
    # here -- you type them in once, below.
    python3 - "${IBC_CONFIG_DIR}/config.ini" <<'PY'
import re, sys, pathlib
path = pathlib.Path(sys.argv[1])
text = path.read_text()
settings = {
    # Paper, always. Flip this deliberately or not at all.
    "TradingMode": "paper",
    # Accept the API connection from localhost without a dialog.
    "AcceptIncomingConnectionAction": "accept",
    "AllowBlindTrading": "yes",
    # The forced daily restart: let IBC drive it rather than a human.
    # 02:00 is after the CME close and before the next session.
    "AutoRestartTime": "02:00 AM",
    # Do not let the gateway close itself on a schedule; the restart above
    # is the only interruption we want.
    "ClosedownAt": "",
    "ExistingSessionDetectedAction": "primary",
    # Read-only would block every order, including paper ones.
    "ReadOnlyApi": "no",
}
for key, value in settings.items():
    pattern = rf"(?m)^{re.escape(key)}=.*$"
    if re.search(pattern, text):
        text = re.sub(pattern, f"{key}={value}", text)
    else:
        text += f"\n{key}={value}\n"
path.write_text(text)
print("applied unattended defaults to", path)
PY
fi

log "environment for systemd"
# The unit reads these rather than carrying a hardcoded version that goes
# stale the moment IBKR ships a new gateway.
cat > "${IBC_CONFIG_DIR}/ibc.env" <<EOF
# Generated by deploy/install-ibc.sh on $(date -Is). Re-run it after a
# gateway upgrade; TWS_MAJOR_VRSN changes with the version.
TWS_MAJOR_VRSN=${TWS_MAJOR_VRSN}
TWS_PATH=/home/${SERVICE_USER}/Jts
TWS_SETTINGS_PATH=${TWS_SETTINGS_DIR}
IBC_INI=${IBC_CONFIG_DIR}/config.ini
IBC_PATH=${IBC_DIR}
LOG_PATH=/home/${SERVICE_USER}/ibc-logs
EOF
chmod 0644 "${IBC_CONFIG_DIR}/ibc.env"
echo "  wrote ${IBC_CONFIG_DIR}/ibc.env (gateway version ${TWS_MAJOR_VRSN})"

# Prove the path IBC will actually construct exists, here, rather than
# letting it fail inside systemd with no context.
gateway_jars="/home/${SERVICE_USER}/Jts/ibgateway/${TWS_MAJOR_VRSN}/jars"
if [[ ! -d "${gateway_jars}" ]]; then
    echo "expected gateway jars at ${gateway_jars}, which does not exist" >&2
    exit 1
fi
echo "  gateway jars: ${gateway_jars}"

# The config holds a password. Treat it accordingly.
chown root:"${SERVICE_USER}" "${IBC_CONFIG_DIR}/config.ini"
chmod 640 "${IBC_CONFIG_DIR}/config.ini"
install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" -m 750 "${TWS_SETTINGS_DIR}"

cat <<EOF

IB Gateway and IBC are installed. Two things left, and both are yours:

  1. Credentials. Edit ${IBC_CONFIG_DIR}/config.ini and set:

         IbLoginId=<your paper username>
         IbPassword=<your paper password>
         TradingMode=paper

     The file is already root:${SERVICE_USER} 0640 so it is not world
     readable. It is NOT in git and must not go there.

  2. Start it:

         sudo systemctl enable --now ibc
         sudo journalctl -u ibc -f

     First login may trigger a two-factor prompt on your phone. Once the
     gateway is up, paper API is on port 4002.

Then: deltahedger doctor -c configs/es_paper.yaml
EOF
