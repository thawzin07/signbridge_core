"""Audit a separate ASL Alphabet Test ZIP against SignBridge's selected dataset."""
from __future__ import annotations
import argparse, json, zipfile
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps
from prepare_dataset import ALL_SYMBOLS, BKTree, dhash, file_sha256, hamming, pixel_sha256
from preprocessing import CLASSES

EXTERNAL_URL = "https://www.kaggle.com/datasets/danrasband/asl-alphabet-test"
EXPECTED_COUNT = 870
EXPECTED_PER_SYMBOL = 30

def label_for(name: str) -> str | None:
    parts = tuple(p for p in name.replace("\\", "/").split("/") if p)
    known = set(ALL_SYMBOLS) | {"DELETE"}
    def canonical(value: str) -> str:
        label = value.upper()
        return "DEL" if label == "DELETE" else label
    for i, part in enumerate(parts):
        if part.lower() == "asl_alphabet_test":
            stem = Path(parts[-1]).stem.upper()
            if stem.endswith("_TEST"):
                stem = stem[:-5]
            candidates = [canonical(candidate) for candidate in parts[i + 1:-1]]
            recognized = [candidate for candidate in candidates if candidate in known]
            if recognized:
                return recognized[-1]
            return canonical(stem)
    if len(parts) >= 2:
        candidate = canonical(parts[-2])
        if candidate in known:
            return candidate
    stem = Path(parts[-1]).stem.upper() if parts else ""
    if stem.endswith("_TEST"):
        stem = stem[:-5]
    return canonical(stem)

def read_pixels(zf: zipfile.ZipFile, member: str):
    with zf.open(member) as handle:
        image = ImageOps.exif_transpose(Image.open(handle)).convert("RGB")
        image.load()
    return image

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("external_zip", type=Path)
    parser.add_argument("prepared_zip", type=Path)
    parser.add_argument("--output", type=Path, default=Path("datasets/external/external_audit.json"))
    parser.add_argument("--near-radius", type=int, default=1)
    parser.add_argument("--license-card-checked", action="store_true", help="Confirm the current Kaggle card reports CC0: Public Domain.")
    args = parser.parse_args()

    prepared_exact = {}
    prepared_tree = BKTree()
    prepared_hashes = []
    prepared_count_by_label = Counter()
    with zipfile.ZipFile(args.prepared_zip) as prepared:
        for info in prepared.infolist():
            if info.is_dir() or not info.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            parts = info.filename.split("/")
            if len(parts) < 4 or parts[0] != "images" or parts[1] not in ("train", "validation", "test"):
                continue
            label = parts[2].upper()
            image = read_pixels(prepared, info.filename)
            exact = pixel_sha256(image)
            prepared_exact.setdefault(exact, []).append((info.filename, label))
            phash = dhash(image)
            index = len(prepared_hashes)
            prepared_tree.add(phash, index)
            prepared_hashes.append((info.filename, label, phash))
            prepared_count_by_label[label] += 1

    external_counts = Counter()
    dimensions = Counter()
    corrupt = []
    exact_matches = []
    near_matches = []
    external_internal_exact = Counter()
    with zipfile.ZipFile(args.external_zip) as external:
        members = []
        for info in external.infolist():
            if info.is_dir() or not info.filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            label = label_for(info.filename)
            if label is None:
                continue
            external_counts[label] += 1
            members.append((info.filename, label))
        for member, label in members:
            try:
                image = read_pixels(external, member)
                dimensions[f"{image.width}x{image.height}"] += 1
                exact = pixel_sha256(image)
                external_internal_exact[exact] += 1
                if exact in prepared_exact:
                    exact_matches.append({"external_member": member, "label": label, "prepared": prepared_exact[exact]})
                phash = dhash(image)
                candidates = prepared_tree.query(phash, args.near_radius)
                for candidate_index in candidates:
                    prepared_member, prepared_label, prior_hash = prepared_hashes[candidate_index]
                    near_matches.append({
                        "external_member": member,
                        "external_label": label,
                        "prepared_member": prepared_member,
                        "prepared_label": prepared_label,
                        "hamming_distance": hamming(phash, prior_hash),
                    })
            except Exception as exc:
                corrupt.append({"member": member, "error": f"{type(exc).__name__}: {exc}"})

    expected_symbols = set(ALL_SYMBOLS)
    blockers = []
    total = sum(external_counts.values())
    if total != EXPECTED_COUNT:
        blockers.append(f"Expected {EXPECTED_COUNT} images from the data card, measured {total}.")
    if set(external_counts) != expected_symbols:
        blockers.append(f"External class names differ from the 29 listed symbols; missing={sorted(expected_symbols-set(external_counts))}, extra={sorted(set(external_counts)-expected_symbols)}.")
    if any(count != EXPECTED_PER_SYMBOL for count in external_counts.values()):
        blockers.append("One or more symbols do not contain the 30 images described by the data card.")
    if dimensions != Counter({"200x200": EXPECTED_COUNT}):
        blockers.append(f"Measured image dimensions differ from 870 200x200 RGB photos: {dict(dimensions)}.")
    if corrupt:
        blockers.append(f"{len(corrupt)} external images could not be decoded.")
    if exact_matches:
        blockers.append(f"{len(exact_matches)} exact cross-dataset image matches were found.")
    if near_matches:
        blockers.append(f"{len(near_matches)} perceptual-hash overlap candidates were found at Hamming radius {args.near_radius}; review them before treating the archives as independent.")
    if not args.license_card_checked:
        blockers.append("The operator has not confirmed the current Kaggle listing license.")
    if any(count > 1 for count in external_internal_exact.values()):
        blockers.append("Exact duplicate images were found within the external archive.")

    result = {
        "external_source": {"title":"ASL Alphabet Test","url":EXTERNAL_URL,"license_card":"CC0: Public Domain","license_card_checked_by_operator":bool(args.license_card_checked)},
        "external_archive": {"name":args.external_zip.name,"size_bytes":args.external_zip.stat().st_size,"sha256":file_sha256(args.external_zip)},
        "prepared_archive": {"name":args.prepared_zip.name,"size_bytes":args.prepared_zip.stat().st_size,"sha256":file_sha256(args.prepared_zip)},
        "measured": {"images":total,"counts_by_label":dict(sorted(external_counts.items())),"dimensions":dict(sorted(dimensions.items())),"corrupt_count":len(corrupt),"internal_exact_duplicate_hashes":sum(1 for count in external_internal_exact.values() if count>1)},
        "overlap": {"exact_match_count":len(exact_matches),"exact_matches":exact_matches,"near_pair_count":len(near_matches),"near_matches":near_matches,"near_duplicate_hamming_radius":args.near_radius},
        "external_eval_ready":not blockers,
        "blockers":blockers,
        "review_limitations":["Archive structure and overlap do not validate ASL semantic labels.","External performance must be reported separately from the prepared test split.","Perceptual-hash overlap candidates are not proof of duplicated content and need review."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"external_eval_ready":result["external_eval_ready"],"images":total,"counts_by_label":result["measured"]["counts_by_label"],"dimensions":result["measured"]["dimensions"],"exact_matches":len(exact_matches),"near_pairs":len(near_matches),"blockers":blockers,"report":str(args.output)},indent=2))
    return 0 if result["external_eval_ready"] else 3

if __name__ == "__main__":
    raise SystemExit(main())
