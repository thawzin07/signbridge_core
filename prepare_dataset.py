"""Audit the Kaggle ASL Alphabet archive and create a reproducible 20-class subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterable

from PIL import Image, ImageDraw, ImageOps

from preprocessing import CLASSES, PREPROCESSING_CONFIG

SOURCE_URL = "https://www.kaggle.com/datasets/grassknoted/asl-alphabet"
SOURCE_DOI = "10.34740/KAGGLE/DSV/29550"
ALL_SYMBOLS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["DEL", "NOTHING", "SPACE"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def dhash(image: Image.Image) -> int:
    pixels = list(ImageOps.grayscale(image).resize((9, 8), Image.Resampling.LANCZOS).getdata())
    value = 0
    for row in range(8):
        for col in range(8):
            value = (value << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class BKTree:
    def __init__(self) -> None:
        self.root: tuple[int, int, dict[int, object]] | None = None

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = (value, index, {})
            return
        node = self.root
        while True:
            distance = hamming(value, node[0])
            child = node[2].get(distance)
            if child is None:
                node[2][distance] = (value, index, {})
                return
            node = child  # type: ignore[assignment]

    def query(self, value: int, radius: int) -> list[int]:
        if self.root is None:
            return []
        found: list[int] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = hamming(value, node[0])
            if distance <= radius:
                found.append(node[1])
            low, high = distance - radius, distance + radius
            stack.extend(child for edge, child in node[2].items() if low <= edge <= high)
        return found


def class_from_source_member(member_name: str) -> tuple[str | None, str | None]:
    parts = tuple(part for part in member_name.replace("\\", "/").split("/") if part)
    source_labels = set(ALL_SYMBOLS) | {"DEL", "DELETE"}
    for index, part in enumerate(parts):
        lower = part.lower()
        if lower == "asl_alphabet_train" and index + 1 < len(parts):
            # The Kaggle ZIP nests a second asl_alphabet_train directory before
            # the 29 class directories. Find the recognized class component,
            # rather than treating that container directory as the label.
            candidates = [candidate.upper() for candidate in parts[index + 1:-1]]
            recognized = [candidate for candidate in candidates if candidate in source_labels]
            if recognized:
                return "train", recognized[-1]
            if candidates:
                return "train", candidates[-1]
        if lower == "asl_alphabet_test" and index + 1 < len(parts):
            # Test images are named A_test.jpg, B_test.jpg, etc., sometimes
            # inside a second asl_alphabet_test container directory.
            stem = Path(parts[-1]).stem.upper()
            if stem.endswith("_TEST"):
                stem = stem[:-5]
            if stem in source_labels:
                return "test", stem
            candidates = [candidate.upper() for candidate in parts[index + 1:-1]]
            recognized = [candidate for candidate in candidates if candidate in source_labels]
            if recognized:
                return "test", recognized[-1]
            return "test", stem
    return None, None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}x3:".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def format_counts(counter: Counter[str], labels: Iterable[str]) -> str:
    return ", ".join(f"{label}={counter.get(label, 0)}" for label in labels)


def make_contact_sheet(rows: list[dict[str, object]], archive: zipfile.ZipFile, output_path: Path, per_class: int = 4) -> None:
    representatives: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if len(representatives[str(row["label"])]) < per_class:
            representatives[str(row["label"])].append(row)
    tile_w, tile_h, header_h = 150, 176, 26
    cols = 4
    rows_per_class = per_class
    sheet = Image.new("RGB", (cols * tile_w, len(CLASSES) * rows_per_class * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    row_num = 0
    for label in CLASSES:
        for sample_num, row in enumerate(representatives.get(label, [])):
            with archive.open(str(row["source_member"])) as handle:
                image = Image.open(handle).convert("RGB")
            image.thumbnail((tile_w - 8, tile_h - header_h - 8), Image.Resampling.LANCZOS)
            col = sample_num % cols
            y = row_num * tile_h
            x = col * tile_w
            draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline=(190, 190, 190))
            draw.text((x + 5, y + 5), f"{label} · sample {sample_num + 1}", fill=(20, 40, 50))
            sheet.paste(image, (x + (tile_w - image.width) // 2, y + header_h + (tile_h - header_h - image.height) // 2))
        row_num += rows_per_class
    sheet.save(output_path, format="PNG", optimize=True)


def prepare(source_zip: Path, output_dir: Path, per_class: int, seed: int, near_radius: int, source_version: str) -> dict[str, object]:
    started = time.monotonic()
    source_zip = source_zip.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_size = source_zip.stat().st_size
    print(f"Hashing source archive ({source_size:,} bytes)...", flush=True)
    source_archive_sha = file_sha256(source_zip)
    print(f"Scanning image members: {source_zip.name}", flush=True)

    train_counts: Counter[str] = Counter()
    test_counts: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    corrupt: list[str] = []
    selected_source_rows: list[dict[str, object]] = []
    total_train_images = 0
    total_test_images = 0
    with zipfile.ZipFile(source_zip) as archive:
        for info in archive.infolist():
            if info.is_dir() or Path(info.filename).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            split, label = class_from_source_member(info.filename)
            if split is None or label is None:
                continue
            if split == "train":
                total_train_images += 1
                train_counts[label] += 1
            else:
                total_test_images += 1
                test_counts[label] += 1
            if (total_train_images + total_test_images) % 10000 == 0:
                print(f"Scanned {total_train_images + total_test_images:,} image members...", flush=True)
            try:
                with archive.open(info) as member_handle:
                    image = Image.open(member_handle)
                    image.load()
                    dimensions[f"{image.width}x{image.height}"] += 1
                    modes[image.mode] += 1
                    if split == "train" and label in CLASSES:
                        rgb = ImageOps.exif_transpose(image).convert("RGB")
                        selected_source_rows.append({
                            "source_member": info.filename,
                            "label": label,
                            "width": image.width,
                            "height": image.height,
                            "pixel_sha256": pixel_sha256(rgb),
                            "dhash": dhash(rgb),
                        })
            except Exception as exc:
                corrupt.append(f"{info.filename}: {type(exc).__name__}: {exc}")

        if not train_counts:
            raise ValueError("No train images found. Expected an asl_alphabet_train/<class>/... layout.")
        missing = sorted(set(CLASSES) - set(train_counts))
        if missing:
            raise ValueError(f"Selected classes are missing from the source archive: {missing}")

        # Exact and perceptual-near duplicates form indivisible groups before sampling/splitting.
        print(f"Grouping same-label near-duplicate candidates among {len(selected_source_rows):,} selected-class source images...", flush=True)
        union = UnionFind(len(selected_source_rows))
        trees_by_label = {label: BKTree() for label in CLASSES}
        for index, row in enumerate(selected_source_rows):
            tree = trees_by_label[str(row["label"])]
            for prior in tree.query(int(row["dhash"]), near_radius):
                union.union(index, prior)
            tree.add(int(row["dhash"]), index)
            if (index + 1) % 10000 == 0:
                print(f"Grouped {index + 1:,} images...", flush=True)
        grouped_indices: dict[int, list[int]] = defaultdict(list)
        for index in range(len(selected_source_rows)):
            grouped_indices[union.find(index)].append(index)

        pixel_groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(selected_source_rows):
            pixel_groups[str(row["pixel_sha256"])].append(index)
        exact_pixel_conflicts = []
        exact_pixel_conflict_indices: set[int] = set()
        for pixel_hash, indices in pixel_groups.items():
            labels = sorted({str(selected_source_rows[index]["label"]) for index in indices})
            if len(labels) > 1:
                exact_pixel_conflict_indices.update(indices)
                exact_pixel_conflicts.append({
                    "pixel_sha256": pixel_hash,
                    "labels": labels,
                    "members": sorted(str(selected_source_rows[index]["source_member"]) for index in indices),
                })

        class_groups: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
        for indices in grouped_indices.values():
            labels = sorted({str(selected_source_rows[index]["label"]) for index in indices})
            members = sorted(str(selected_source_rows[index]["source_member"]) for index in indices)
            group_id = hashlib.sha256("\n".join(members).encode("utf-8")).hexdigest()[:16]
            if any(index in exact_pixel_conflict_indices for index in indices):
                continue
            label = labels[0]
            class_groups[label].append((group_id, indices))

        rng = random.Random(seed)
        selected: list[dict[str, object]] = []
        class_available: dict[str, int] = {}
        for label in CLASSES:
            groups = class_groups.get(label, [])
            rng.shuffle(groups)
            class_available[label] = len(groups)
            for group_id, indices in groups[:per_class]:
                # One deterministic representative per group; members of each group stay together.
                representative = min(indices, key=lambda index: (str(selected_source_rows[index]["pixel_sha256"]), str(selected_source_rows[index]["source_member"])))
                selected.append({**selected_source_rows[representative], "group_id": group_id})

        split_by_group: dict[str, str] = {}
        for label in CLASSES:
            class_rows = [row for row in selected if row["label"] == label]
            rng.shuffle(class_rows)
            n = len(class_rows)
            train_end = int(round(n * 0.8))
            validation_end = train_end + int(round(n * 0.1))
            for index, row in enumerate(class_rows):
                split_name = "train" if index < train_end else "validation" if index < validation_end else "test"
                split_by_group[str(row["group_id"])] = split_name

        manifest_rows: list[dict[str, object]] = []
        selected_images: list[tuple[str, bytes]] = []
        print(f"Encoding {len(selected):,} selected subset images...", flush=True)
        for selected_index, row in enumerate(selected, start=1):
            label = str(row["label"])
            group_id = str(row["group_id"])
            split_name = split_by_group[group_id]
            archive_path = f"images/{split_name}/{label}/{str(row['pixel_sha256'])[:12]}.jpg"
            with archive.open(str(row["source_member"])) as member_handle:
                image = ImageOps.exif_transpose(Image.open(member_handle)).convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=95, optimize=True, progressive=False)
                encoded = buffer.getvalue()
            prepared_hash = hashlib.sha256(encoded).hexdigest()
            manifest_rows.append({
                "archive_path": archive_path,
                "split": split_name,
                "label": label,
                "class_index": CLASSES.index(label),
                "group_id": group_id,
                "source_member": row["source_member"],
                "source_pixel_sha256": row["pixel_sha256"],
                "prepared_file_sha256": prepared_hash,
                "dhash_hex": f"{int(row['dhash']):016x}",
                "source_width": row["width"],
                "source_height": row["height"],
            })
            selected_images.append((archive_path, encoded))
            if selected_index % 2000 == 0:
                print(f"Encoded {selected_index:,}/{len(selected):,} images...", flush=True)

        manifest_path = output_dir / "manifest.csv"
        fields = list(manifest_rows[0].keys()) if manifest_rows else []
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(manifest_rows)

        class_mapping = {label: index for index, label in enumerate(CLASSES)}
        mapping_path = output_dir / "class_to_index.json"
        mapping_path.write_text(json.dumps({"classes": CLASSES, "class_to_index": class_mapping}, indent=2) + "\n", encoding="utf-8")
        preprocess_path = output_dir / "preprocessing.json"
        preprocess_path.write_text(json.dumps(PREPROCESSING_CONFIG, indent=2) + "\n", encoding="utf-8")

        dataset_zip = output_dir / "selected_asl_alphabet_20class.zip"
        with zipfile.ZipFile(dataset_zip, "w", compression=zipfile.ZIP_STORED) as prepared_archive:
            for archive_path, encoded in selected_images:
                prepared_archive.writestr(archive_path, encoded)
            prepared_archive.write(manifest_path, "manifest.csv")
            prepared_archive.write(mapping_path, "class_to_index.json")
            prepared_archive.write(preprocess_path, "preprocessing.json")

        selected_counts = Counter(str(row["label"]) for row in manifest_rows)
        split_counts = {split_name: Counter(str(row["label"]) for row in manifest_rows if row["split"] == split_name) for split_name in ("train", "validation", "test")}
        class_counts = {
            label: {split_name: int(split_counts[split_name].get(label, 0)) for split_name in ("train", "validation", "test")}
            for label in CLASSES
        }
        groups_by_split: dict[str, set[str]] = {
            split_name: {str(row["group_id"]) for row in manifest_rows if row["split"] == split_name}
            for split_name in ("train", "validation", "test")
        }
        split_leakage_groups = sorted(
            (groups_by_split["train"] & groups_by_split["validation"])
            | (groups_by_split["train"] & groups_by_split["test"])
            | (groups_by_split["validation"] & groups_by_split["test"])
        )
        # A contact sheet is a review aid only and does not imply human signoff.
        contact_rows = sorted(manifest_rows, key=lambda row: (int(row["class_index"]), row["source_pixel_sha256"]))
        contact_sheet = Image.new("RGB", (8 * 150, 10 * 176), "white")
        draw = ImageDraw.Draw(contact_sheet)
        sheet_index = 0
        contact_per_class: Counter[str] = Counter()
        for row in contact_rows:
            label = str(row["label"])
            rank = contact_per_class[label]
            if rank >= 4:
                continue
            contact_per_class[label] += 1
            with archive.open(str(row["source_member"])) as image_handle:
                image = ImageOps.exif_transpose(Image.open(image_handle)).convert("RGB")
            image.thumbnail((142, 142), Image.Resampling.LANCZOS)
            col, sheet_row = sheet_index % 8, sheet_index // 8
            x, y = col * 150, sheet_row * 176
            draw.rectangle((x, y, x + 149, y + 175), outline=(190, 190, 190))
            draw.text((x + 5, y + 5), f"{label} · sample {rank + 1}", fill=(20, 40, 50))
            contact_sheet.paste(image, (x + (150 - image.width) // 2, y + 30 + (142 - image.height) // 2))
            sheet_index += 1
        contact_path = output_dir / "review_contact_sheet.png"
        contact_sheet.save(contact_path, format="PNG", optimize=True)

        duration = round(time.monotonic() - started, 2)
        prepared_archive_sha = file_sha256(dataset_zip)
        provenance = {
            "primary_source": {
                "title": "ASL Alphabet",
                "url": SOURCE_URL,
                "publisher": "Akash Nagaraj (Kaggle account grassknoted)",
                "citation_doi": SOURCE_DOI,
                "license": "GPL-2 (as listed on Kaggle when checked 2026-10-07)",
                "source_version": source_version,
                "source_archive_file": source_zip.name,
                "source_archive_size_bytes": source_size,
                "source_archive_size_mib": round(source_size / (1024 * 1024), 3),
                "source_archive_sha256": source_archive_sha,
                "listing_claims": {"training_images": 87000, "dimensions": "200x200", "classes": 29},
                "measured_contents": {
                    "train_images": total_train_images,
                    "test_images": total_test_images,
                    "train_counts_by_class": dict(sorted(train_counts.items())),
                    "test_counts_by_class": dict(sorted(test_counts.items())),
                    "image_dimensions": dict(sorted(dimensions.items())),
                    "image_modes": dict(sorted(modes.items())),
                    "corrupt_image_count": len(corrupt),
                    "corrupt_images": corrupt,
                },
                "attribution": f"Akash Nagaraj, ASL Alphabet, Kaggle, DOI {SOURCE_DOI}.",
            },
            "prepared_subset": {
                "classes": CLASSES,
                "target_per_class": per_class,
                "actual_counts": dict(sorted(selected_counts.items())),
                "split_counts": class_counts,
                "archive_file": dataset_zip.name,
                "archive_size_bytes": dataset_zip.stat().st_size,
                "archive_sha256": prepared_archive_sha,
                "duplicate_grouping": {"method": "64-bit difference-hash connected components are built within each configured label to keep candidate groups from chaining across visually similar classes. Cross-label exclusion uses exact RGB pixel hashes only; perceptual similarity is not proof of a duplicate.", "near_duplicate_hamming_threshold": near_radius, "same_label_candidate_group_count": int(sum(1 for members in grouped_indices.values() if len(members) > 1)), "exact_pixel_duplicate_excess": int(sum(len(indices) - 1 for indices in pixel_groups.values() if len(indices) > 1)), "cross_label_exact_pixel_conflict_group_count": len(exact_pixel_conflicts), "excluded_cross_label_exact_pixel_image_count": len(exact_pixel_conflict_indices), "cross_label_exact_pixel_conflicts": exact_pixel_conflicts},
                "random_seed": seed,
                "available_groups_by_class": class_available,
                "split_leakage_group_count": len(split_leakage_groups),
                "preparation_seconds": duration,
            },
            "review_status": "pending_asl_competent_review",
            "technical_audit_passed": not bool(corrupt) and not exact_pixel_conflicts and bool(manifest_rows) and all(selected_counts.get(label, 0) == per_class for label in CLASSES) and not split_leakage_groups,
            "audit_ready_for_colab": False,
            "audit_readiness_note": "ASL-competent semantic review is still pending; an experimental training override does not represent reviewer approval.",
            "technical_audit_note": "Same-label perceptual candidate groups are kept within one split. Cross-label copies are excluded only when normalized RGB pixels match exactly. Technical readiness requires clean decoding, no exact cross-label duplicates, the configured balanced sample for every class, and no group leakage across splits.",
            "signer_session_metadata": "No trustworthy signer/session IDs were found in the archive layout; grouped splits cannot establish unseen-signer performance.",
            "small_source_test_set": "Kaggle's supplied 29-image test folder is counted and audited but is not used as the project test split.",
            "external_holdout": "Not audited unless a separate external archive is explicitly supplied to the external-audit script.",
        }
        provenance_path = output_dir / "dataset_provenance.json"
        provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        audit_lines = [
            "# SignBridge Core dataset audit",
            "",
            f"- Source archive: `{source_zip.name}` ({source_size:,} bytes / {source_size / (1024 * 1024):.3f} MiB).",
            f"- Source SHA-256: `{source_archive_sha}`.",
            f"- Kaggle version supplied to the command: `{source_version}`.",
            f"- Measured source contents: {total_train_images:,} train images, {total_test_images:,} source-test images, {len(corrupt)} corrupt images.",
            f"- Dimensions observed: `{json.dumps(dict(sorted(dimensions.items())), sort_keys=True)}`.",
            f"- Selected subset: {sum(selected_counts.values()):,} images across the exact 20 configured classes.",
            f"- Prepared archive: `{dataset_zip.name}` ({dataset_zip.stat().st_size:,} bytes), SHA-256 `{prepared_archive_sha}`.",
            f"- Near-duplicate candidate rule: 64-bit difference-hash connected components are built within each configured label (Hamming radius {near_radius}) so similar-looking classes do not chain together. Cross-label exclusions use exact RGB pixel hashes only; {len(exact_pixel_conflicts)} cross-label exact-pixel groups ({len(exact_pixel_conflict_indices)} images) were excluded.",
            "- Split: deterministic, approximately stratified 80/10/10 by selected image group, before augmentation.",
            "- Signer/session IDs: unavailable; this split does not establish performance on unseen signers.",
            "- Kaggle's tiny source test folder was measured but not used as the project test set.",
            "- Human review: **pending ASL-competent review**. The contact sheet is an inspection aid, not reviewer approval.",
            f"- `technical_audit_passed`: `{provenance['technical_audit_passed']}` (requires clean decoding, the configured balanced count in every class, and no group leakage).",
            "- `audit_ready_for_colab`: `false` because ASL-competent semantic review is pending.",
            f"- Preparation time: {duration:.2f} seconds.",
            "",
            "## Class counts",
            "",
            "| Letter | Train | Validation | Test | Total |",
            "|---|---:|---:|---:|---:|",
        ]
        for label in CLASSES:
            counts = class_counts[label]
            total = sum(counts.values())
            audit_lines.append(f"| {label} | {counts['train']} | {counts['validation']} | {counts['test']} | {total} |")
        audit_lines.extend(["", "## Original class counts", "", "| Source class | Train files | Source-test files |", "|---|---:|---:|"])
        for label in sorted(set(train_counts) | set(test_counts)):
            audit_lines.append(f"| {label} | {train_counts.get(label, 0)} | {test_counts.get(label, 0)} |")
        (output_dir / "audit_report.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_zip", nargs="?", type=Path, default=Path("datasets/source/asl-alphabet.zip"))
    parser.add_argument("--output", type=Path, default=Path("datasets/prepared"))
    parser.add_argument("--per-class", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--near-radius", type=int, default=1)
    parser.add_argument("--source-version", default="unknown; supply verified Kaggle dataset version")
    args = parser.parse_args()
    if not args.source_zip.is_file():
        parser.error(f"Source archive not found: {args.source_zip}")
    if args.per_class < 1 or not 0 <= args.near_radius <= 8:
        parser.error("--per-class must be positive and --near-radius must be between 0 and 8.")
    result = prepare(args.source_zip, args.output, args.per_class, args.seed, args.near_radius, args.source_version)
    print(json.dumps({
        "source_archive_size_bytes": result["primary_source"]["source_archive_size_bytes"],
        "source_archive_sha256": result["primary_source"]["source_archive_sha256"],
        "prepared_counts": result["prepared_subset"]["actual_counts"],
        "prepared_archive": result["prepared_subset"]["archive_file"],
        "audit_ready_for_colab": result["audit_ready_for_colab"],
        "human_review_status": result["review_status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
