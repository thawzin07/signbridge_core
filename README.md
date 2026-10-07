# SignBridge Core

SignBridge Core is a small, local demo that classifies **static images** into these 20 configured ASL alphabet labels:

`A B C D E F G H I K L O P Q R U V W X Y`

The scope is one static letter per frame. It does not cover J or Z, dynamic signing, fingerspelling sequences, words, or sentence translation. A prediction is a model output, not a language interpretation.

## Current project status

- Source archive `datasets/source/asl-alphabet.zip`: 1,100,887,034 bytes; SHA-256 `7c572f14fbaff94f98835cfe71c7582dd379a5176e7c4f83dbf3a30e4b3f68c4`.
- Measured source: 87,000 training images, 28 source-test images, all 200×200, and 0 corrupt images. Each of the 29 source training folders contained 3,000 images.
- Prepared subset: 12,000 RGB images, 600 per configured class. Split counts are 480 train, 60 validation, and 60 test images per class. The archive is 147,134,857 bytes; SHA-256 `c7e15631d28abbcff06ffe12d8ca982b24336dc4a62b7b62db37a63006483021`.
- The technical data audit passed: all configured classes have the balanced sample, images decode, manifest hashes match, and near-duplicate groups do not cross splits. Same-label difference-hash groups use Hamming radius 1. There were no exact RGB duplicates across labels. Perceptual cross-label similarity is not treated as a duplicate because those hashes can chain across visually different images.
- ASL-competent semantic label review remains **pending**, so `audit_ready_for_colab` is `false`. The contact sheet is a review aid and does not count as approval.
- Colab training completed on 2026-10-07. `models/best_model.keras` is the selected 20-class MobileNetV2 artifact, with labels and run metadata. The saved held-out CSV test result is 99.0% accuracy and 0.990 macro-F1 over 1,200 prepared images. These results describe that dataset split only; webcam accuracy and semantic label correctness remain unverified.
- The local app and model-loading path have passed a one-image smoke test using the prepared test archive. A live webcam session still requires manual testing on the target computer.

These status statements should be updated from measured artifacts after the archive is available locally; do not replace them with estimates.

## Project layout

```text
signbridge_core/
  app.py                         Local Gradio webcam and uploaded-image demo
  hand_presence.py               MediaPipe hand-presence gate
  inference.py                   Model loading and prediction helpers
  preprocessing.py               Shared crop, RGB, resize, and normalization rules
  prepare_dataset.py              Source ZIP audit and deterministic subset builder
  audit_external_holdout.py       Optional independent ZIP overlap audit
  split_prepared_archive.py       Makes checksum-verified Drive transfer parts
  build_notebook.py               Rebuilds the Colab training notebook
  requirements.txt                Local application and preparation dependencies
  datasets/
    source/                       Original Kaggle ZIP (not committed)
    prepared/                     Selected subset and audit outputs
  notebooks/                      Colab training notebook
  models/                         Trained model and metadata sidecars
  reports/                        Training and evaluation outputs
  vercel_app/                     Public browser demo and serverless inference endpoint
```

## 1. Prepare and audit the Kaggle archive

Copy the ZIP from Drive into `datasets/source/` and name the local copy `asl-alphabet.zip`. Keep the source ZIP unchanged. Use Python 3.12 for the local GUI environment.

From this project directory, run:

```powershell
python prepare_dataset.py datasets/source/asl-alphabet.zip `
  --output datasets/prepared `
  --per-class 600 `
  --seed 20261007 `
  --near-radius 1
```

If the Kaggle revision is visible and can be verified from the source page or download record, add `--source-version "<verified version>"`. Otherwise leave it recorded as unknown. The script hashes the source and output, checks image decoding and dimensions, reports label counts, groups likely duplicates before splitting, and writes a review contact sheet. The configured sample count is a cap; the output report records actual counts.

Review these outputs before training:

- `datasets/prepared/audit_report.md`
- `datasets/prepared/dataset_provenance.json`
- `datasets/prepared/review_contact_sheet.png`
- `datasets/prepared/manifest.csv`

The source card reported the ASL Alphabet dataset under GPL-2 when checked on 2026-10-07. Confirm the source card and applicable terms again before redistribution. Dataset listing facts are not a substitute for inspection of the downloaded archive.

## 2. Train in Colab

Place the manifest, class map, preprocessing configuration, provenance JSON, `transport_shards.json`, and both prepared archive parts in `My Drive/SignBridge_Core/datasets/prepared/`. The Drive connector limits each uploaded file to 100 MiB, so the notebook concatenates the two parts and checks their hashes against the prepared archive before extracting it. Place `preprocessing.py` at the `SignBridge_Core/` project root, and open `notebooks/signbridge_core_training.ipynb` in Colab.

The notebook checks the archive and manifest, keeps grouped train/validation/test splits, trains MobileNetV2, chooses its rejection threshold using validation data, evaluates the held-out test split once, and exports the model and reports to Drive. The notebook contains an explicit experimental-training override because semantic review is pending. This permits an exploratory run; it does not approve labels or establish general ASL performance.

Keep training metrics scoped to the measured split. The source archive does not provide verified signer/session identities, so grouped image splits cannot establish unseen-signer performance. A model reload smoke check is not a webcam test.

## 3. Run the local demo

The Windows virtual environment for this checkout is at the workspace root in `v/`; keeping it short avoids Windows path-length failures inside TensorFlow. From the workspace root, recreate it with Python 3.12 if needed and install the pinned dependencies:

```powershell
py -3.12 -m venv v
.\v\Scripts\python.exe -m pip install -r ".\project demonstration preparation\signbridge_core\requirements.txt"
```

The selected Colab model and its sidecars are already in `models/`. The local hand-presence gate uses Google's MediaPipe Hand Landmarker task asset stored there as `hand_landmarker.task`. The camera capture loop continuously keeps the newest frame, while a separate worker checks new frames about four times per second at most; this prevents a slow CPU prediction from blocking the preview. TensorFlow CPU thread use is capped at four by default and can be changed with `SIGNBRIDGE_TF_THREADS`. To start the app from the SignBridge Core folder, run:

```powershell
..\..\v\Scripts\python.exe app.py
```

Open the local URL printed by Gradio (configured for `127.0.0.1:7860`). The live tab uses the computer's webcam and provides Start/Stop controls, a configurable crop, hand-presence gating, and a short temporal vote. The image tab accepts a still image with the same crop controls. The displayed score is a softmax score, not calibrated certainty.

## 4. Verify the camera flow on the target computer

After installing dependencies and copying a trained model, check each of the following manually:

1. Start the app and grant camera access if Windows prompts.
2. Confirm a hand in frame produces a crop and a prediction; move the hand and check that the live preview updates.
3. Remove the hand and confirm the app reports no hand instead of retaining a new letter prediction.
4. Stop and restart the camera; confirm that capture actually stops and restarts.
5. Upload a still image and compare its crop with the live crop controls.
6. Try letter capture, Backspace, Space, and Clear. Treat the resulting string as a sequence of classifier outputs, not a recognized word.

Record the computer, camera, model hash, and observed result in `reports/` if this flow is demonstrated. Do not describe code inspection, dataset evaluation, or a model reload as camera validation.

## Public web demo

`vercel_app/` is a separate browser-based demo for public testing. It keeps camera preview local to the browser, then sends compressed still snapshots to a TensorFlow CPU function in the configured Singapore region for inference. A default-off acknowledgement is required before camera or photo inference, and the function processes images in memory without writing image files. Because images leave the browser for prediction, do not test with sensitive photos. The public endpoint is suitable for a small class demonstration; it is not a production service and has no global abuse-prevention layer.

Public test URL: [https://signbridge-live-demo.vercel.app/](https://signbridge-live-demo.vercel.app/)

The web demo reports a four-of-five-frame stable letter and adds characters to the sequence only after a manual click. Its single-image upload uses the same crop and inference path. It does not perform word recognition. Read `vercel_app/README.md` for run and evidence boundaries. A public URL should be called live-camera tested only after manual testing on the intended browser and camera.

## Optional independent holdout audit

If a separately sourced external ZIP is available and its current source card has been checked, run:

```powershell
python audit_external_holdout.py path\to\external.zip `
  datasets/prepared/selected_asl_alphabet_20class.zip `
  --output datasets/external/external_audit.json `
  --license-card-checked
```

The external audit checks archive shape and overlap against the prepared subset. It does not verify sign semantics, signer independence, or population representativeness.
