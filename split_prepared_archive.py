"""Split the prepared ZIP into verified sub-100-MiB Drive transfer parts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_archive(source: Path, output_dir: Path, part_size_mib: int = 90) -> dict[str, object]:
    source = source.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if part_size_mib < 1 or part_size_mib > 95:
        raise ValueError("part_size_mib must be between 1 and 95 MiB to stay below the Drive upload limit.")

    part_bytes = part_size_mib * 1024 * 1024
    source_hash = sha256_file(source)
    parts: list[dict[str, object]] = []
    with source.open("rb") as source_handle:
        part_number = 1
        while True:
            output_path = output_dir / f"{source.name}.part-{part_number:02d}"
            digest = hashlib.sha256()
            bytes_written = 0
            with output_path.open("wb") as output_handle:
                while bytes_written < part_bytes:
                    chunk = source_handle.read(min(1024 * 1024, part_bytes - bytes_written))
                    if not chunk:
                        break
                    output_handle.write(chunk)
                    digest.update(chunk)
                    bytes_written += len(chunk)
            if bytes_written == 0:
                output_path.unlink(missing_ok=True)
                break
            parts.append({
                "file": output_path.name,
                "size_bytes": bytes_written,
                "sha256": digest.hexdigest(),
            })
            print(f"Wrote {output_path.name}: {bytes_written:,} bytes", flush=True)
            part_number += 1

    metadata: dict[str, object] = {
        "archive_file": source.name,
        "archive_size_bytes": source.stat().st_size,
        "archive_sha256": source_hash,
        "part_size_mib": part_size_mib,
        "parts": parts,
        "reassembly": "Concatenate the parts in listed order as raw bytes, then open the result as a ZIP archive.",
    }
    manifest_path = output_dir / "transport_shards.json"
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path, default=Path("datasets/prepared/selected_asl_alphabet_20class.zip"))
    parser.add_argument("--output", type=Path, default=Path("datasets/prepared"))
    parser.add_argument("--part-size-mib", type=int, default=90)
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"Prepared archive not found: {args.archive}")
    metadata = split_archive(args.archive, args.output, args.part_size_mib)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
