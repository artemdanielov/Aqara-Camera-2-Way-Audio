"""go2rtc stream registration for the Aqara camera backchannel.

Registers a go2rtc stream with three producers: an RTSP source for the
camera video/audio, an ffmpeg source that re-exposes the stream with
Opus audio for WebRTC compatibility, and an exec source that runs
bridge.py to ferry the WebRTC mic audio back to the camera speaker.

Registration is done by writing to go2rtc.yaml because go2rtc blocks
exec sources via REST API for security. The integration continues to
work without go2rtc -- only the WebRTC backchannel is affected.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import TypeGuard
from urllib.parse import quote

import yaml

from .const import RTSP_PORT

_LOGGER = logging.getLogger(__name__)

GO2RTC_CONFIG_FILE = "go2rtc.yaml"
_LEGACY_EXEC_SUFFIX = "#backchannel=1"
_EXEC_SUFFIX = "#backchannel=1#audio=pcma/8000#killsignal=15#killtimeout=10"

# Serialises read-modify-write on go2rtc.yaml across threads (the integration
# dispatches register_stream/remove_stream via hass.async_add_executor_job, so
# concurrent setup of two camera entries would otherwise race).
_CONFIG_LOCK = threading.Lock()


class _QuotedDumper(yaml.SafeDumper):
    # Empty subclass so add_representer below does not mutate the global
    # SafeDumper state shared with the rest of the process.
    pass


def _quoted_str_representer(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    if "#" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_QuotedDumper.add_representer(str, _quoted_str_representer)


def _dump_config(config: dict) -> str:
    """Dump go2rtc config to YAML, quoting strings that contain #."""
    return yaml.dump(config, Dumper=_QuotedDumper, default_flow_style=False, width=4096)


def _read_config(config_path: Path) -> dict:
    """Read go2rtc.yaml, returning {} if absent. Atomic (no exists() probe)."""
    try:
        raw = config_path.read_text()
    except FileNotFoundError:
        return {}
    config = yaml.safe_load(raw)
    return {} if config is None else config


def _write_config(config_path: Path, config: dict) -> None:
    config_path.write_text(_dump_config(config))


def build_exec_source(config_dir: str, camera_ip: str) -> str:
    """Build the go2rtc exec source URL for the backchannel bridge.

    The exec source tells go2rtc to spawn the bridge script when a browser
    microphone connects. Explicit PCMA/8kHz matches the bridge's A-law
    decoder. SIGTERM lets its existing cleanup send STOP_VOICE, with a
    ten-second graceful-stop wait limit on go2rtc builds supporting it.

    go2rtc uses exec.Command (not a shell), so we pass the absolute script
    path. go2rtc's ParseQuery splits on # (not &) for multiple parameters.
    """
    # Defense in depth: the exec source is space-delimited, so any whitespace
    # in camera_ip would inject extra argv elements into bridge.py.
    assert not any(c.isspace() for c in camera_ip), "camera_ip must not contain whitespace"
    python = sys.executable
    bridge_script = str(
        Path(config_dir)
        / "custom_components"
        / "aqara_camera_2_way_audio"
        / "bridge.py"
    )
    return f"exec:{python} {bridge_script} {camera_ip}{_EXEC_SUFFIX}"


def build_rtsp_source(
    camera_ip: str, username: str, password: str, channel: int = 1
) -> str:
    """Build the RTSP source URL for the camera stream on a given channel."""
    user = quote(username, safe="")
    pwd = quote(password, safe="")
    return f"rtsp://{user}:{pwd}@{camera_ip}:{RTSP_PORT}/ch{channel}"


def register_stream(
    config_dir: str,
    stream_name: str,
    sources: list[str],
) -> str:
    """Add a stream to go2rtc.yaml.

    Returns one of:
      - "wrote": new content written
      - "noop": exact sources or equivalent bridge options already in the file
      - "skipped": a stream by this name exists with different sources NOT
        matching ours -- treated as a collision with another integration
        or another camera entry; do not overwrite
      - "error": IO or parse failure (logged, never raised)
    """
    config_path = Path(config_dir) / GO2RTC_CONFIG_FILE
    try:
        if not _valid_sources(sources):
            raise ValueError("Invalid requested source types")
        with _CONFIG_LOCK:
            config = _read_config(config_path)
            if not isinstance(config, dict):
                raise ValueError("Invalid config type")
            streams = config.setdefault("streams", {})
            if not isinstance(streams, dict):
                raise ValueError("Invalid streams type")
            existing = streams.get(stream_name)

            if stream_name in streams:
                if _valid_sources(existing) and Counter(existing) == Counter(sources):
                    _LOGGER.debug("go2rtc stream %s already current", stream_name)
                    return "noop"
                # Retain equivalent option orderings without rewriting YAML or
                # treating them as ownership for a subsequent update.
                if _is_retained_customization(existing, sources):
                    _LOGGER.debug("go2rtc stream %s bridge options retained", stream_name)
                    return "noop"
                # Refuse to clobber if the existing entry is plausibly not
                # ours. An entry whose exec source does not match ours
                # belongs to another camera (or another integration).
                if (
                    not _valid_sources(existing)
                    or sum(s.startswith("exec:") for s in existing) != 1
                    or sum(s.startswith("exec:") for s in sources) != 1
                    or not _stream_belongs_to_us(existing, sources, config_dir)
                ):
                    _LOGGER.warning(
                        "go2rtc stream %s already exists with different sources "
                        "(possible collision). Not overwriting.",
                        stream_name,
                    )
                    return "skipped"

            streams[stream_name] = sources
            _write_config(config_path, config)
        _LOGGER.info(
            "go2rtc stream '%s' written to %s. Restart HA for backchannel to take effect.",
            stream_name,
            config_path,
        )
        return "wrote"

    except Exception as err:
        # YAML exceptions can embed credential-bearing source lines.
        _LOGGER.warning("Failed to write go2rtc config (%s)", type(err).__name__)
        return "error"


def _valid_sources(sources: object) -> TypeGuard[list[str]]:
    """Validate source types before comparisons or string operations."""
    return isinstance(sources, list) and all(isinstance(s, str) for s in sources)


def _is_retained_customization(existing: object, sources: list[str]) -> bool:
    """Retain only equivalent options on the same command and other sources."""
    if not _valid_sources(existing):
        return False
    our_exec = [s for s in sources if s.startswith("exec:")]
    old_exec = [s for s in existing if s.startswith("exec:")]
    if len(our_exec) != 1 or len(old_exec) != 1:
        return False
    command, separator, params = our_exec[0].partition("#")
    if separator != "#" or "#" + params != _EXEC_SUFFIX:
        return False
    old_command, separator, old_params = old_exec[0].partition("#")
    if separator != "#" or old_command != command:
        return False
    # Compare raw tokens, rejecting duplicate, unknown or altered options.
    if Counter(old_params.split("#")) != Counter(params.split("#")):
        return False
    return Counter(s for s in existing if not s.startswith("exec:")) == Counter(
        s for s in sources if not s.startswith("exec:")
    )


def _stream_belongs_to_us(
    existing_sources: list[str], our_sources: list[str], config_dir: str
) -> bool:
    """Heuristic: same stream name belongs to us if our exec source matches.

    The exec source encodes the bridge.py absolute path and the camera IP,
    making it a strong ownership signature.
    """
    our_exec = next((s for s in our_sources if s.startswith("exec:")), None)
    if our_exec is None:
        return False
    if our_exec in existing_sources:
        return True

    # Upgrade only the exact former output of this integration's builder.
    # Rebuild against this config directory and interpreter before accepting
    # the legacy suffix; never strip options from arbitrary caller commands.
    camera_ip = our_exec.rsplit(" ", 1)[-1].partition("#")[0]
    if not camera_ip or any(c.isspace() for c in camera_ip):
        return False
    if our_exec != build_exec_source(config_dir, camera_ip):
        return False
    legacy_exec = our_exec.removesuffix(_EXEC_SUFFIX) + _LEGACY_EXEC_SUFFIX
    return legacy_exec in existing_sources


def remove_stream(config_dir: str, stream_name: str) -> None:
    """Remove a stream from go2rtc.yaml. Best-effort, never raises."""
    config_path = Path(config_dir) / GO2RTC_CONFIG_FILE

    try:
        with _CONFIG_LOCK:
            config = _read_config(config_path)
            streams = config.get("streams", {})
            if stream_name not in streams:
                return
            del streams[stream_name]
            _write_config(config_path, config)
        _LOGGER.info("go2rtc stream '%s' removed from %s", stream_name, config_path)
    except Exception:
        _LOGGER.warning("Failed to update go2rtc config", exc_info=True)
