# Forward walk on a VPS

A runbook for getting the GEX straddle system paper-trading unattended.
Read the whole thing once before running any of it — two of these steps are
irreversible in ways worth knowing about first.

**None of this has been run against a real VPS.** The scripts are
syntax-checked and the systemd units pass `systemd-analyze verify`, but the
IBKR download URLs move and the first run on your box is the first run
anywhere. Expect to fix one or two things, and use `deltahedger doctor` to
tell you which.

## What you need

- A VPS with Ubuntu 22.04+ or Debian 12+ (systemd, `apt`), 2 GB RAM minimum —
  IB Gateway is a Java desktop app and wants about 1 GB on its own.
- An **IBKR paper account** with CME futures market data. The username and
  password go on the box; plan for that before you start.
- The market-data permission that carries **generic tick 101 (option open
  interest)**. Without it there is no GEX, and the strategy stands aside on
  every bar — correctly, but you will have deployed a system that does
  nothing. `doctor` checks this explicitly because it is the failure most
  likely to waste a week.
- Timezone sanity: the exchange calendar is America/New_York and the code
  converts, but a VPS whose clock is wrong will enter at the wrong time.
  Check `timedatectl` and make sure NTP is on.

## 1. Bootstrap

```bash
git clone https://github.com/vivere7108-lab/DeltaHedgedShortVol.git /tmp/dh
sudo /tmp/dh/deploy/bootstrap.sh
```

Creates a `deltahedger` service user, clones to `/opt/deltahedger`, builds a
venv, installs the systemd units. Starts nothing. Re-runnable — it is how you
deploy an update later.

Override with environment variables if your layout differs:

```bash
sudo INSTALL_DIR=/srv/dh SERVICE_USER=trader /tmp/dh/deploy/bootstrap.sh
```

## 2. IB Gateway

```bash
sudo /opt/deltahedger/deploy/install-ibc.sh
```

This installs IB Gateway and [IBC](https://github.com/IbcAlpha/IBC), which
drives it. IBC is not optional for an unattended run: the gateway expects a
human to type a password, and **IBKR forces it to restart every day**. IBC
handles both. Without it your walk dies at the first restart and the
strategy log looks completely healthy while it happens.

Then put your credentials in:

```bash
sudo nano /etc/ibc/config.ini
```

```ini
IbLoginId=<your paper username>
IbPassword=<your paper password>
TradingMode=paper
```

The file is `root:deltahedger` mode 0640, so it is not world-readable. **It
is not in git and must never go there.** If you would rather not have a
password on disk at all, IBC can prompt on first start instead — but then a
reboot needs you present, which defeats the point.

```bash
sudo systemctl enable --now ibc
sudo journalctl -u ibc -f
```

First login usually triggers **two-factor on your phone**. Approve it. IBKR
re-prompts periodically (roughly weekly); when the walk goes quiet, this is
the first thing to check.

### If git says "detected dubious ownership"

```
fatal: detected dubious ownership in repository at '/opt/deltahedger'
```

Git refuses to act on a repository owned by someone else, and `bootstrap.sh`
deliberately hands `/opt/deltahedger` to the service user — so a bare
`sudo git pull` as root hits this every time.

**Use `bootstrap.sh` to update, not `git pull`.** It runs git as the repo's
owner and prints the commit it landed on:

```bash
sudo /opt/deltahedger/deploy/bootstrap.sh
#   now at ea7ccf3 Fix IB Gateway and IBC install failing with Permission denied
```

The trap to know about: a failed pull leaves the checkout *silently stale*,
and a stale script fails in exactly the same way as an unpatched one. If a
fix does not appear to have worked, check what you are actually running
before re-diagnosing the bug — `install-ibc.sh` prints its own commit on
every run for this reason:

```
install-ibc.sh from ea7ccf3 2026-09-02 Fix IB Gateway and IBC install ...
```

If you must use git directly, run it as the owner:

```bash
sudo -u deltahedger git -C /opt/deltahedger pull
```

### If the installer says "Permission denied"

```
sudo: unable to execute /tmp/tmp.XXXX/ibgateway.sh: Permission denied
```

Three unrelated causes produce that exact message, which is why the script
now handles all three rather than guessing:

1. **The directory, not the file.** `mktemp -d` run as root creates a
   `0700 root` directory. The service user cannot traverse into it, so the
   script's own `0755` is irrelevant. Staging now happens under
   `/home/<service-user>/.install` instead.
2. **`chmod u+x` on a root-owned file.** `unzip` under umask 022 leaves
   scripts `0644`; `chmod u+x` makes that `0744` — runnable by root and by
   nobody else. IBC's launcher is now `0755`, which matters because
   `ibc.service` runs as the service user and would otherwise have failed at
   `ExecStart` with the same message and no context.
3. **`/tmp` mounted `noexec`**, common on hardened VPS images. Staging off
   `/tmp` avoids it for the download, and `TMPDIR` is pointed at the staging
   directory so the install4j bundle unpacks somewhere it is allowed to run.

`install-ibc.sh` now verifies executability as the service user and stops
with the offending mode printed, rather than letting the failure surface
three steps later inside systemd. If you hit it anyway:

```bash
stat -c '%a %U:%G %n' /opt/ibc/gatewaystart.sh
namei -l /opt/ibc/gatewaystart.sh        # every directory on the path
mount | grep -E ' /tmp | /home '         # noexec?
```

### If nothing is listening on port 4002

`doctor` now tells you which case you are in, because "connection refused"
does not distinguish a gateway still starting from one that died at launch:

```bash
systemctl status ibc --no-pager -l
journalctl -u ibc -n 80 --no-pager
```

- **Service active, port closed.** The gateway takes 30–90 seconds to
  launch, log in and open the API port — longer if two-factor is waiting on
  your phone. Watch `journalctl -u ibc -f` and re-run `doctor`.
- **Service failed or inactive.** Usually credentials not set in
  `/etc/ibc/config.ini`, or an unapproved 2FA prompt on first login.
- **Version mismatch.** IBC locates the gateway at
  `$TWS_PATH/ibgateway/$TWS_MAJOR_VRSN/jars`. That version is *detected* by
  `install-ibc.sh` and written to `/etc/ibc/ibc.env`; re-run the installer
  after a gateway upgrade. Check what it found:

  ```bash
  cat /etc/ibc/ibc.env
  ls /home/deltahedger/Jts/ibgateway/        # should be a version number
  ```

  If that lists `jars` rather than a version number like `1030`, the install
  is in the flat layout IBC cannot use — re-run `install-ibc.sh`, which
  moves it aside and reinstalls correctly.

## 3. Preflight — the important step

```bash
sudo -u deltahedger /opt/deltahedger/.venv/bin/deltahedger doctor \
    -c /opt/deltahedger/configs/es_paper.yaml
```

```
  [ok  ] connect to IBKR                  127.0.0.1:4002
  [ok  ] account is paper                 DU1234567 (paper)
  [ok  ] qualified the future             ESU5
  [ok  ] qualified the hedge              MESU5
  [ok  ] future price                     5,412.25
  [ok  ] a 0DTE series is listed          2025-09-02
  [ok  ] ATM straddle quote               5410 @ 21.40 (IV 0.118)
  [ok  ] open interest (generic tick 101) 41 strikes, 38,204 contracts
  [ok  ] journal directory is writable    /opt/deltahedger/runs/live

GEX +412.7M/1% at 5,412.25, flip 5,388.0, regime positive
  right now it would: SHORT the ATM straddle and collect theta
```

Every one of those is a thing you would otherwise discover at 09:35 as the
strategy quietly doing nothing — which in a log is indistinguishable from
the market genuinely reading neutral. **Do not skip to step 4 with a
failing check.**

Run it again after any gateway restart or config change. It places no
orders and is safe to run at any time.

## 4. Start the walk — in dry-run first

```bash
sudo systemctl enable --now deltahedger
sudo journalctl -u deltahedger -f
```

It ships with `--dry-run`: every decision computed and logged, no order
placed. **Leave it there for at least one full session.** A dry run
exercises the entire path except the fill, which is the cheapest way to find
out that your entry window doesn't match how the day actually looks, or that
the regime never leaves neutral on real open interest, or that the ATM
straddle is wider than you expected.

Read what it did:

```bash
sudo -u deltahedger /opt/deltahedger/.venv/bin/deltahedger report \
    -c /opt/deltahedger/configs/es_paper.yaml --show-events
```

When you are satisfied, drop the flag:

```bash
sudo nano /etc/deltahedger.env      # remove --dry-run
sudo systemctl restart deltahedger
```

That routes orders to the **paper** account. `es_paper.yaml` sets
`ibkr.allow_live_trading: true`, because `cli.py`'s `live` command refuses
to route *any* non-dry-run order -- paper included -- without it; the flag
is not solely a live-account switch. With it set, the account actually used
is decided by which account the Gateway session is logged into
(`TradingMode` in `/etc/ibc/config.ini`), not by this file, so keep that
pointed at paper until the walk has earned a live account.

## What runs, and what it survives

```
  ibc.service ──── IB Gateway under Xvfb, auto-login, daily self-restart
       │
       │ API on 127.0.0.1:4002 (paper)
       ▼
  deltahedger.service ──── polls every 5s, journals every decision
       │
       ▼
  /opt/deltahedger/runs/live/{events,fills,bars}-YYYY-MM-DD.jsonl
```

The two are ordered but **not bound**. The runner reconnects on its own with
exponential backoff, so the daily gateway restart costs it a few seconds and
nothing else; binding the units would restart the strategy every night for
an event it already handles.

| Failure | What happens |
|---|---|
| Daily gateway restart | Runner reconnects, backing off 15s → 300s. Logged. |
| Network blip | Same path. |
| Gateway crashes | systemd restarts `ibc` after 60s; runner reconnects when it returns. |
| Strategy process dies | systemd restarts it after 30s; it re-reconciles positions against the broker rather than trusting a stale book. |
| VPS reboot | Both units are enabled, so both come back. IBKR may want 2FA. |
| Journal write fails | Logged as an error; **trading continues**. Losing the log must not take the position with it. |
| An option position it did not open | Refuses to start. Adopting a half-known straddle is how a book ends up long gamma while the strategy believes it is short it. |

Restart limits are deliberate on both units (5 starts per 10 minutes). A
login loop against IBKR gets the account locked, which is much worse than
being down for a few minutes.

## Watching it

```bash
sudo journalctl -u deltahedger -f                       # live decisions
sudo journalctl -u ibc --since today                    # gateway health
deltahedger report -c configs/es_paper.yaml             # session summary
deltahedger report -c configs/es_paper.yaml --day 2025-09-02 --show-events
```

The runner logs a heartbeat every 5 minutes even when nothing happens, so a
quiet log and a stalled process can be told apart.

The journal is JSON Lines and append-only. A restart mid-session adds to the
day's files rather than truncating them, so an interrupted walk loses the
position but never the history. Analyse it with the same tools as a
backtest:

```python
from deltahedger.live.journal import read_journal
bars = read_journal("runs/live", "bars")      # every poll, all sessions
events = read_journal("runs/live", "events")
```

## Updating

```bash
sudo /opt/deltahedger/deploy/bootstrap.sh     # pulls as the owner, reinstalls
sudo systemctl restart deltahedger
```

Not `git pull` as root — see "detected dubious ownership" above. Bootstrap
prints the commit it landed on, so a stale checkout is visible rather than
inferred.

Do it outside market hours. A restart mid-session is safe — positions are
re-read from the broker — but it drops the in-memory session P&L baselines
the position-P&L exits are measured against, so an open long straddle would
have its stop reset.

## Before you read anything into the results

The README's "Reading a result honestly" section applies here too, and one
part of it applies *more*. A forward walk is a single path. The system's
discrete hedging leaves a few percent of path-dependent residual per run,
which is real and mean-zero — so a fortnight of paper trading is a check
that the machinery works in the real world, **not** a measurement of edge.
What it can tell you:

- does the regime classification match what the market actually did?
- how often does the flip get crossed, and does exiting on it help?
- what do fills actually cost versus the modelled tick of slippage?
- is the fixed ±10 band sane against live gamma, or does it churn?

Those are the questions the walk answers. Whether the strategy makes money
is not one of them yet.
