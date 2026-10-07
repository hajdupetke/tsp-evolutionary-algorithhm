"""Download TSPLIB datasets into data/ (stdlib only).

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --force
    python scripts/download_data.py --data-dir data berlin52 burma14
    python scripts/download_data.py --timeout 60

Existing valid files are skipped unless --force is given. Each dataset is
tried against multiple mirror URLs (.tsp.gz preferred, plain .tsp fallback)
so the original Heidelberg URL going 404 does not break setup.
"""

from __future__ import annotations

import argparse
import gzip
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

# Expected DIMENSION per TSPLIB header (used for validation).
DATASETS: dict[str, int] = {
    "burma14": 14,
    "berlin52": 52,
    "kroA100": 100,
    "kroA200": 200,
    "pcb442": 442,
    "pr1002": 1002,
}

HEIDELBERG_NEW = "https://www.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/tsp"
HEIDELBERG_OLD = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp"
ZIB = "https://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp"
# GitHub mirrors (verified 2026-10-02: all 4 datasets return valid TSPLIB text).
JORLIB_RAW = (
    "https://raw.githubusercontent.com/coin-or/jorlib/master"
    "/jorlib-core/src/test/resources/tspLib/tsp"
)
TSPLIBNET_RAW = (
    "https://raw.githubusercontent.com/pdrozdowski/TSPLib.Net/master/TSPLIB95/tsp"
)


def candidate_urls(name: str) -> list[str]:
    """Mirror URLs for one dataset, most reliable first.

    GitHub raw mirrors first (verified, plain .tsp); official Heidelberg
    (.tsp.gz preferred, plain .tsp fallback) and ZIB mirrors last — the
    Heidelberg site is intermittently down (verified 2026-10-02).
    """
    return [
        f"{JORLIB_RAW}/{name}.tsp",
        f"{TSPLIBNET_RAW}/{name}.tsp",
        f"{HEIDELBERG_NEW}/{name}.tsp.gz",
        f"{HEIDELBERG_OLD}/{name}.tsp.gz",
        f"{ZIB}/{name}.tsp.gz",
        f"{HEIDELBERG_NEW}/{name}.tsp",
        f"{HEIDELBERG_OLD}/{name}.tsp",
        f"{ZIB}/{name}.tsp",
    ]


def fetch_bytes(url: str, timeout: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "tsp-ga-setup/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def decode_payload(raw: bytes, url: str) -> str:
    if url.endswith(".gz"):
        try:
            raw = gzip.decompress(raw)
        except OSError as e:
            raise ValueError(f"invalid gzip payload from {url}: {e}") from e
    return raw.decode("utf-8", errors="strict")


def validate_content(name: str, text: str) -> None:
    expected_dim = DATASETS[name]
    if "NODE_COORD_SECTION" not in text:
        raise ValueError("missing NODE_COORD_SECTION (not a TSPLIB .tsp file?)")
    for line in text.splitlines():
        if line.strip().startswith("DIMENSION"):
            dim = int(line.split(":")[1].strip())
            if dim != expected_dim:
                raise ValueError(f"DIMENSION={dim}, expected {expected_dim}")
            return
    raise ValueError("missing DIMENSION header")


def is_valid_existing(path: Path, name: str) -> bool:
    if not path.is_file():
        return False
    try:
        validate_content(name, path.read_text(encoding="utf-8"))
        return True
    except (OSError, ValueError, UnicodeDecodeError):
        return False


def download_one(name: str, data_dir: Path, timeout: int, force: bool) -> Path:
    dest = data_dir / f"{name}.tsp"
    if not force and is_valid_existing(dest, name):
        print(f"[skip] {dest} already exists and looks valid. Use --force to re-download.")
        return dest

    errors: list[str] = []
    for url in candidate_urls(name):
        try:
            print(f"[try] {name} <- {url}")
            text = decode_payload(fetch_bytes(url, timeout), url)
            validate_content(name, text)
            data_dir.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
            print(f"[ok] wrote {dest} ({len(text)} chars)")
            return dest
        except Exception as e:  # noqa: BLE001 - try next mirror
            errors.append(f"{url}: {e}")
            print(f"[fail] {url}: {e}")

    raise RuntimeError(f"All mirrors failed for {name}:\n  " + "\n  ".join(errors))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download TSPLIB .tsp files into data/.")
    p.add_argument("datasets", nargs="*", default=None,
                   choices=sorted(DATASETS),
                   help="Which datasets to fetch (default: all).")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                   help="Destination directory (default: <repo>/data).")
    p.add_argument("--timeout", type=int, default=30,
                   help="Per-request timeout in seconds (default: 30).")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if a valid file already exists.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    datasets = args.datasets or sorted(DATASETS)
    failed: list[str] = []
    for name in datasets:
        try:
            download_one(name, data_dir, args.timeout, args.force)
        except Exception as e:  # noqa: BLE001 - report all, exit 1
            print(f"[error] {name}: {e}", file=sys.stderr)
            failed.append(name)
    if failed:
        print(f"Failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("All datasets ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
