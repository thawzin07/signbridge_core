"""Generate the executable SignBridge Core Colab notebook with only Python's standard library."""
import json
from pathlib import Path

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def py(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}

cells = [
md("""# SignBridge Core: static ASL fingerspelling letters

This notebook trains a 20-class classifier for A, B, C, D, E, F, G, H, I, K, L, O, P, Q, R, U, V, W, X, Y. J and Z are excluded because their forms are motion-dependent. It recognizes one static letter at a time and does not recognize words, translate sentences, or understand sign language.

The prepared split is approximately stratified 80/10/10 and keeps detected exact/near-duplicate groups together. The source has no trustworthy signer/session IDs, so this split cannot establish unseen-signer performance. ASL-competent label review remains pending. The explicit experimental override below permits exploratory training without changing that review status.

The shared preprocessing helper defines RGB order, EXIF orientation, square crop, preserved aspect ratio, TensorFlow bilinear resize to 160x160, MobileNetV2 normalization to [-1, 1], and the class order. The local GUI imports the same helper. The setup below supports Colab's Python 3.13 runtime and checks compatible package ranges rather than exact patch releases.
"""),
py("""%pip install -q "tensorflow>=2.21.0,<2.22.0" "scikit-learn>=1.6.1,<1.7.0" "matplotlib>=3.10.0,<3.11.0" "Pillow>=11.1.0,<12.0.0" """),
py("""import os, sys, json, time, hashlib, zipfile, shutil, platform, random, csv
from pathlib import Path
from collections import Counter
import numpy as np
import tensorflow as tf
import sklearn, matplotlib, PIL
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

SUPPORTED = {"tensorflow":">=2.21.0,<2.22.0", "scikit-learn":">=1.6.1,<1.7.0", "matplotlib":">=3.10.0,<3.11.0", "Pillow":">=11.1.0,<12.0.0"}
ACTUAL = {"tensorflow":tf.__version__, "scikit-learn":sklearn.__version__, "matplotlib":matplotlib.__version__, "Pillow":PIL.__version__}
print("Python:", sys.version, "| platform:", platform.platform())
print("Supported / actual package versions:", json.dumps({"supported":SUPPORTED,"actual":ACTUAL}, indent=2))
incompatible = {name: version for name, version in ACTUAL.items() if Version(version) not in SpecifierSet(SUPPORTED[name])}
if incompatible: raise RuntimeError(f"Packages outside supported ranges: {incompatible}. Rerun the dependency setup cell, then restart the runtime.")
gpus = tf.config.list_physical_devices("GPU")
print("GPU devices:", gpus)
for gpu in gpus:
    try: tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError: pass
tf.keras.mixed_precision.set_global_policy("mixed_float16" if gpus else "float32")
if not gpus: print("No GPU found. Training may take longer; the time budget may stop before fine-tuning completes.")
SEED = 20261007
random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
try: tf.config.experimental.enable_op_determinism()
except Exception: pass
"""),
md("""## Mount Drive and find prepared files

Upload the prepared archive parts, transport_shards.json, manifest, class map, and provenance file under SignBridge_Core/datasets/prepared. Upload preprocessing.py to the SignBridge_Core project root. The notebook verifies and concatenates the parts into the exact prepared ZIP in Colab-local /content storage.
"""),
py("""from google.colab import drive
drive.mount("/content/drive")
DRIVE_ROOT = Path("/content/drive/MyDrive/SignBridge_Core")
DATA_DIR = DRIVE_ROOT / "datasets"
PREPARED_DIR = DATA_DIR / "prepared"
MODEL_DIR = DRIVE_ROOT / "models"
REPORT_DIR = DRIVE_ROOT / "reports"
MANIFEST_PATH = PREPARED_DIR / "manifest.csv"
MAPPING_PATH = PREPARED_DIR / "class_to_index.json"
PROVENANCE_PATH = PREPARED_DIR / "dataset_provenance.json"
SHARD_MANIFEST_PATH = PREPARED_DIR / "transport_shards.json"
HELPER_PATH = DRIVE_ROOT / "preprocessing.py"
for path in (MANIFEST_PATH, MAPPING_PATH, PROVENANCE_PATH, SHARD_MANIFEST_PATH, HELPER_PATH):
    if not path.is_file(): raise FileNotFoundError(f"Missing uploaded project file: {path}")
transport = json.loads(SHARD_MANIFEST_PATH.read_text(encoding="utf-8"))
provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
prepared_meta = provenance["prepared_subset"]
if transport.get("archive_sha256") != prepared_meta.get("archive_sha256") or transport.get("archive_size_bytes") != prepared_meta.get("archive_size_bytes"):
    raise ValueError("Drive shard manifest does not match the prepared dataset provenance.")
DATASET_ZIP = Path("/content/signbridge_core_prepared.zip")
assembled_digest = hashlib.sha256(); assembled_size = 0
with DATASET_ZIP.open("wb") as output_zip:
    for part in transport["parts"]:
        relative_part = Path(part["file"])
        if relative_part.is_absolute() or ".." in relative_part.parts:
            raise ValueError(f"Unsafe dataset shard path: {relative_part}")
        shard_path = PREPARED_DIR / relative_part
        if not shard_path.is_file(): raise FileNotFoundError(shard_path)
        part_digest = hashlib.sha256(); part_size = 0
        with shard_path.open("rb") as source_part:
            for chunk in iter(lambda: source_part.read(1024 * 1024), b""):
                output_zip.write(chunk); part_digest.update(chunk); assembled_digest.update(chunk)
                part_size += len(chunk); assembled_size += len(chunk)
        if part_size != part["size_bytes"] or part_digest.hexdigest() != part["sha256"]:
            raise ValueError(f"Drive dataset shard checksum/size mismatch: {shard_path.name}")
if assembled_size != transport["archive_size_bytes"] or assembled_digest.hexdigest() != transport["archive_sha256"]:
    raise ValueError("Reassembled dataset ZIP checksum/size does not match its transport manifest.")
print("Reassembled and SHA-256 verified prepared archive:", DATASET_ZIP, "| bytes:", assembled_size)
MODEL_DIR.mkdir(parents=True, exist_ok=True); REPORT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(DRIVE_ROOT))
from preprocessing import CLASSES, PREPROCESSING_CONFIG, tf_preprocess_tensor, tf_resize_rgb_tensor
expected_classes = ["A","B","C","D","E","F","G","H","I","K","L","O","P","Q","R","U","V","W","X","Y"]
assert CLASSES == expected_classes and len(CLASSES) == 20
print("Class order:", CLASSES)
print("Preprocessing:", json.dumps(PREPROCESSING_CONFIG, indent=2))
"""),
md("""## Extract and validate data before fitting

The original 29-image Kaggle test folder is not used. This preparation uses its own grouped train/validation/test split.
"""),
py("""extract_started = time.monotonic()
EXTRACT_DIR = Path("/content/signbridge_core_data")
if EXTRACT_DIR.exists(): shutil.rmtree(EXTRACT_DIR)
EXTRACT_DIR.mkdir(parents=True)
with zipfile.ZipFile(DATASET_ZIP) as archive:
    bad_member = archive.testzip()
    if bad_member: raise RuntimeError(f"Archive CRC failed for {bad_member}")
    for info in archive.infolist():
        member = Path(info.filename)
        if member.is_absolute() or ".." in member.parts: raise RuntimeError(f"Unsafe archive member: {info.filename}")
    archive.extractall(EXTRACT_DIR)
EXTRACT_SECONDS = time.monotonic() - extract_started
with MANIFEST_PATH.open(encoding="utf-8", newline="") as handle: manifest = list(csv.DictReader(handle))
class_map_file = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
class_to_index = class_map_file["class_to_index"]
assert class_map_file["classes"] == CLASSES
assert class_to_index == {letter:i for i,letter in enumerate(CLASSES)}
assert manifest and {row["label"] for row in manifest} == set(CLASSES)
assert {row["split"] for row in manifest} == {"train","validation","test"}
assert provenance.get("technical_audit_passed") is True, "Technical data audit must pass before training."

ALLOW_UNREVIEWED_EXPERIMENTAL_TRAINING = True
if provenance.get("review_status") != "approved_by_asl_competent_reviewer":
    print("REVIEW LIMITATION: semantic label review is pending. This run is experimental and does not imply approval.")
    if not ALLOW_UNREVIEWED_EXPERIMENTAL_TRAINING:
        raise RuntimeError("Enable the explicit experimental override only if you intend exploratory training.")

split_counts = {name:Counter() for name in ("train","validation","test")}
group_splits, hash_splits = {}, {}
paths_by_split = {name:[] for name in split_counts}
labels_by_split = {name:[] for name in split_counts}
for row in manifest:
    path = EXTRACT_DIR / row["archive_path"]
    if not path.is_file(): raise FileNotFoundError(path)
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if file_hash != row["prepared_file_sha256"]: raise ValueError(f"Image checksum mismatch: {path.name}")
    split, label, group = row["split"], row["label"], row["group_id"]
    if group in group_splits and group_splits[group] != split: raise ValueError(f"Near-duplicate group crosses splits: {group}")
    if file_hash in hash_splits and hash_splits[file_hash] != split: raise ValueError("Exact duplicate crosses splits.")
    group_splits[group] = split; hash_splits[file_hash] = split
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB": raise ValueError(f"Image is not RGB: {path}")
        if min(image.size) < 32: raise ValueError(f"Image is too small for MobileNetV2: {path}")
    split_counts[split][label] += 1
    paths_by_split[split].append(str(path))
    labels_by_split[split].append(class_to_index[label])
for split in split_counts:
    missing = sorted(set(CLASSES) - set(split_counts[split]))
    if missing: raise ValueError(f"{split} split is missing classes: {missing}")
print(f"Extracted in {EXTRACT_SECONDS:.1f}s; images={len(manifest)}")
print("Split counts:", {split:dict(counts) for split,counts in split_counts.items()})
print("Source license:", provenance["primary_source"]["license"], "| label review:", provenance["review_status"])
"""),
py("""# Visual spot-check uses training examples only. It does not approve semantic labels.
fig, axes = plt.subplots(4, 5, figsize=(13, 11))
for index, letter in enumerate(CLASSES):
    row = next(item for item in manifest if item["split"]=="train" and item["label"]==letter)
    with Image.open(EXTRACT_DIR / row["archive_path"]) as image: axes.flat[index].imshow(image.convert("RGB"))
    axes.flat[index].set_title(letter); axes.flat[index].axis("off")
plt.suptitle("One training image per class — inspect labels and image quality")
plt.tight_layout(); plt.show()
"""),
md("""## Efficient tf.data pipeline and conservative augmentation

Images are decoded as RGB, cropped to the largest square, resized through the shared helper, then augmented on training batches only. Horizontal flips are omitted. The model receives MobileNetV2-normalized float32 RGB.
"""),
py("""AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32
train_paths = np.asarray(paths_by_split["train"]); train_labels = np.asarray(labels_by_split["train"], dtype=np.int32)
val_paths = np.asarray(paths_by_split["validation"]); val_labels = np.asarray(labels_by_split["validation"], dtype=np.int32)
test_paths = np.asarray(paths_by_split["test"]); test_labels = np.asarray(labels_by_split["test"], dtype=np.int32)

def decode_rgb(path, label):
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    return tf_resize_rgb_tensor(image), label

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.025, fill_mode="reflect", seed=SEED),
    tf.keras.layers.RandomTranslation(0.035, 0.035, fill_mode="reflect", seed=SEED+1),
    tf.keras.layers.RandomZoom((-0.05,0.05), (-0.05,0.05), fill_mode="reflect", seed=SEED+2),
    tf.keras.layers.RandomContrast(0.08, seed=SEED+3),
], name="conservative_train_only_augmentation")

def augment_and_normalize(image, label):
    image = augmentation(image, training=True)
    return tf.keras.applications.mobilenet_v2.preprocess_input(image), label

def normalize_only(image, label):
    return tf.keras.applications.mobilenet_v2.preprocess_input(image), label

train_ds = tf.data.Dataset.from_tensor_slices((train_paths,train_labels))
train_ds = train_ds.shuffle(min(len(train_paths),12000), seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.map(decode_rgb, num_parallel_calls=AUTOTUNE).map(augment_and_normalize, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((val_paths,val_labels)).map(decode_rgb,num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(normalize_only,num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
print("Counts:", len(train_paths), len(val_paths), len(test_paths), "| batch size:", BATCH_SIZE)
"""),
md("""## MobileNetV2 frozen-head stage

BatchNormalization stays in inference mode. The checkpoint is selected by validation macro-F1; early stopping and learning-rate reduction avoid costly searches. A 45-minute wall-clock callback spans both stages and reports the first-epoch throughput.
"""),
py("""def build_model():
    inputs = tf.keras.Input((160,160,3), dtype=tf.float32, name="rgb_160_normalized")
    backbone = tf.keras.applications.MobileNetV2(input_shape=(160,160,3), include_top=False, weights="imagenet", name="mobilenetv2_backbone")
    backbone.trainable = False
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25, seed=SEED)(x)
    outputs = tf.keras.layers.Dense(20, activation="softmax", dtype="float32", name="letter_probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name="signbridge_core_mobilenetv2")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    return model

class ValidationMacroF1(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        probabilities = self.model.predict(val_ds, verbose=0)
        score = f1_score(val_labels, probabilities.argmax(axis=1), labels=np.arange(20), average="macro", zero_division=0)
        logs["val_macro_f1"] = float(score)
        print(f" — val_macro_f1: {score:.4f}")

class TrainingTimeBudget(tf.keras.callbacks.Callback):
    def __init__(self, seconds):
        super().__init__(); self.seconds=seconds; self.started=time.monotonic(); self.durations=[]
    def on_epoch_begin(self, epoch, logs=None): self.epoch_started=time.monotonic()
    def on_epoch_end(self, epoch, logs=None):
        duration=time.monotonic()-self.epoch_started; self.durations.append(duration)
        if epoch==0:
            batches=int(np.ceil(len(train_paths)/BATCH_SIZE))
            print(f"Warm-up throughput: {batches/max(duration,1e-6):.2f} batches/s; {len(train_paths)/max(duration,1e-6):.1f} images/s")
        elapsed=time.monotonic()-self.started; remaining=max(0,self.seconds-elapsed)
        average=float(np.mean(self.durations))
        print(f"Budget: {elapsed/60:.1f} min elapsed, {remaining/60:.1f} min remaining; estimated epochs remaining: {int(remaining/max(average,1e-6))}")
        if elapsed>=self.seconds:
            print("Time budget reached; keeping the best saved checkpoint.")
            self.model.stop_training=True

TRAINING_BUDGET_SECONDS=45*60
budget=TrainingTimeBudget(TRAINING_BUDGET_SECONDS)
best_frozen=MODEL_DIR/"best_frozen.keras"
head_started=time.monotonic()
model=build_model()
weight_setup_seconds=time.monotonic()-head_started
head_history=model.fit(
    train_ds, validation_data=val_ds, epochs=30,
    callbacks=[
        ValidationMacroF1(),
        tf.keras.callbacks.ModelCheckpoint(best_frozen,monitor="val_macro_f1",mode="max",save_best_only=True,verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.3,patience=2,min_lr=1e-6,verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1",mode="max",patience=5,restore_best_weights=True,verbose=1),
        tf.keras.callbacks.BackupAndRestore(str(MODEL_DIR/"training_backup_frozen"),save_freq="epoch",delete_checkpoint=False),
        budget,
    ], verbose=1)
HEAD_SECONDS=time.monotonic()-head_started
if not best_frozen.is_file(): raise RuntimeError("No usable frozen-head checkpoint was saved.")
print(f"Frozen-head stage: {HEAD_SECONDS/60:.1f} min; model setup / pretrained weights: {weight_setup_seconds:.1f} s")
"""),
md("""## Fine-tune the upper backbone layers

Fine-tuning uses a lower learning rate. BatchNormalization layers remain non-trainable, and the backbone is called with training=False so moving statistics are not updated.
"""),
py("""best_finetuned=MODEL_DIR/"best_finetuned.keras"
FINETUNE_SECONDS=0.0
if time.monotonic()-budget.started < TRAINING_BUDGET_SECONDS:
    model=tf.keras.models.load_model(best_frozen,compile=False)
    backbone=model.get_layer("mobilenetv2_backbone"); backbone.trainable=True
    for layer in backbone.layers[:-30]: layer.trainable=False
    for layer in backbone.layers:
        if isinstance(layer,tf.keras.layers.BatchNormalization): layer.trainable=False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    fine_started=time.monotonic()
    model.fit(
        train_ds,validation_data=val_ds,epochs=15,
        callbacks=[
            ValidationMacroF1(),
            tf.keras.callbacks.ModelCheckpoint(best_finetuned,monitor="val_macro_f1",mode="max",save_best_only=True,verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.3,patience=2,min_lr=1e-7,verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1",mode="max",patience=4,restore_best_weights=True,verbose=1),
            tf.keras.callbacks.BackupAndRestore(str(MODEL_DIR/"training_backup_finetune"),save_freq="epoch",delete_checkpoint=False),
            budget,
        ],verbose=1)
    FINETUNE_SECONDS=time.monotonic()-fine_started
else:
    print("Fine-tuning skipped because the time budget was reached.")
print(f"Fine-tuning stage: {FINETUNE_SECONDS/60:.1f} min")
"""),
md("""## Select model and rejection threshold on validation data only

The rejection threshold is chosen on known validation classes. A softmax score is not calibrated certainty and cannot guarantee rejection of unsupported signs or arbitrary images.
"""),
py("""def score_checkpoint(path):
    candidate=tf.keras.models.load_model(path,compile=False)
    probabilities=candidate.predict(val_ds,verbose=0)
    score=float(f1_score(val_labels,probabilities.argmax(axis=1),labels=np.arange(20),average="macro",zero_division=0))
    return score,candidate

frozen_score,frozen_model=score_checkpoint(best_frozen)
choices=[(frozen_score,best_frozen)]
if best_finetuned.is_file():
    fine_score,_=score_checkpoint(best_finetuned); choices.append((fine_score,best_finetuned))
best_validation_f1,best_path=max(choices,key=lambda item:item[0])
model=tf.keras.models.load_model(best_path,compile=False)
val_probs=model.predict(val_ds,verbose=0); val_max=val_probs.max(axis=1)
threshold_rows=[]
for threshold in np.linspace(0.10,0.99,90):
    accepted=val_max>=threshold
    predictions=np.where(accepted,val_probs.argmax(axis=1),-1)
    score=f1_score(val_labels,predictions,labels=np.arange(20),average="macro",zero_division=0)
    threshold_rows.append((float(score),float(accepted.mean()),float(threshold)))
_,VALIDATION_COVERAGE,VALIDATION_THRESHOLD=max(threshold_rows,key=lambda item:(item[0],item[1]))
print("Chosen checkpoint:",best_path.name,"validation macro-F1:",round(best_validation_f1,4))
print("Validation threshold:",round(VALIDATION_THRESHOLD,3),"known-class coverage:",round(VALIDATION_COVERAGE,3))
"""),
md("""## Final evaluation on the untouched test split

Report raw classification metrics and thresholded coverage separately. These are dataset results only.
"""),
py("""evaluation_started=time.monotonic()
test_ds=tf.data.Dataset.from_tensor_slices((test_paths,test_labels)).map(decode_rgb,num_parallel_calls=AUTOTUNE)
test_ds=test_ds.map(normalize_only,num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_probs=model.predict(test_ds,verbose=1); test_pred=test_probs.argmax(axis=1); test_score=test_probs.max(axis=1)
accepted=test_score>=VALIDATION_THRESHOLD
rejected_pred=np.where(accepted,test_pred,-1)
precision,recall,class_f1,support=precision_recall_fscore_support(test_labels,test_pred,labels=np.arange(20),zero_division=0)
test_metrics={
 "accuracy":float(accuracy_score(test_labels,test_pred)),
 "macro_f1":float(f1_score(test_labels,test_pred,labels=np.arange(20),average="macro",zero_division=0)),
 "validation_rejection_threshold":float(VALIDATION_THRESHOLD),
 "accepted_coverage":float(accepted.mean()),
 "accepted_only_accuracy":float(accuracy_score(test_labels[accepted],test_pred[accepted])) if accepted.any() else None,
 "rejected_count":int((~accepted).sum()),
 "per_class":{letter:{"precision":float(precision[i]),"recall":float(recall[i]),"f1":float(class_f1[i]),"support":int(support[i])} for i,letter in enumerate(CLASSES)},
 "classification_report":classification_report(test_labels,test_pred,labels=np.arange(20),target_names=CLASSES,output_dict=True,zero_division=0),
 "confusion_matrix_raw":confusion_matrix(test_labels,test_pred,labels=np.arange(20)).tolist(),
 "confusion_matrix_with_reject":confusion_matrix(test_labels,rejected_pred,labels=list(range(20))+[-1]).tolist(),
}
EVALUATION_SECONDS=time.monotonic()-evaluation_started
print(json.dumps({k:v for k,v in test_metrics.items() if k not in ("per_class","classification_report","confusion_matrix_raw","confusion_matrix_with_reject")},indent=2))
print(classification_report(test_labels,test_pred,labels=np.arange(20),target_names=CLASSES,zero_division=0))
"""),
py("""fig,ax=plt.subplots(figsize=(12,10))
cm=np.asarray(test_metrics["confusion_matrix_raw"])
im=ax.imshow(cm,cmap="Blues"); ax.set_xticks(range(20),CLASSES); ax.set_yticks(range(20),CLASSES)
ax.set_xlabel("Predicted letter"); ax.set_ylabel("True dataset label"); ax.set_title("Untouched test split confusion matrix")
fig.colorbar(im,ax=ax); plt.tight_layout(); plt.show()

fig,axes=plt.subplots(1,2,figsize=(12,4))
for key,label in (("loss","train loss"),("val_loss","validation loss")):
    if key in head_history.history: axes[0].plot(head_history.history[key],label=label)
if "val_macro_f1" in head_history.history: axes[1].plot(head_history.history["val_macro_f1"],label="frozen-head validation macro-F1")
axes[0].set_title("Loss"); axes[1].set_title("Validation macro-F1")
for axis in axes: axis.set_xlabel("Epoch"); axis.legend(); axis.grid(alpha=.2)
plt.tight_layout(); plt.show()
"""),
py("""# Highest-scoring test errors, shown after the final test evaluation.
wrong=np.flatnonzero(test_pred!=test_labels)
wrong=sorted(wrong,key=lambda i:float(test_probs[i,test_pred[i]]),reverse=True)[:12]
rows=max(1,int(np.ceil(len(wrong)/4)))
fig,axes=plt.subplots(rows,4,figsize=(12,3*rows)); axes=np.asarray(axes).reshape(-1)
for axis,index in zip(axes,wrong):
    with Image.open(test_paths[index]) as image: axis.imshow(image.convert("RGB"))
    axis.set_title(f"true {CLASSES[test_labels[index]]} -> {CLASSES[test_pred[index]]} ({test_probs[index,test_pred[index]]:.2f})")
    axis.axis("off")
for axis in axes[len(wrong):]: axis.axis("off")
plt.suptitle("Representative test errors; dataset labels have not received ASL-competent signoff")
plt.tight_layout(); plt.show()
"""),
md("""## Export artifacts and run reload/inference smoke test

Exports go to Drive models/ and reports/. The model is not considered webcam-verified by this dataset evaluation.
"""),
py("""import datetime
final_model_path=MODEL_DIR/"best_model.keras"
model.save(final_model_path)
labels_export={"classes":CLASSES,"class_to_index":class_to_index}
preprocess_export={**PREPROCESSING_CONFIG,"image_size":[160,160],"validation_rejection_threshold":float(VALIDATION_THRESHOLD),"score_interpretation":"Softmax model score; not calibrated certainty."}
training_config={
 "architecture":"TensorFlow/Keras MobileNetV2, ImageNet pretrained",
 "input_shape":[160,160,3],"batch_size":BATCH_SIZE,"seed":SEED,
 "frozen_head_max_epochs":30,"fine_tune_max_epochs":15,"fine_tune_upper_layers":30,
 "batch_normalization":"Non-trainable; backbone called with training=False during fine-tuning.",
 "optimizer_head":"Adam 1e-3","optimizer_fine_tune":"Adam 1e-5",
 "wall_clock_budget_seconds":TRAINING_BUDGET_SECONDS,
 "validation_rejection_threshold":float(VALIDATION_THRESHOLD),
 "threshold_selection":"Validation macro-F1 with coverage tie-break; known classes only.",
 "experimental_without_asl_review":provenance.get("review_status")!="approved_by_asl_competent_reviewer",
}
metrics_export={"test":test_metrics,"validation":{"macro_f1":float(best_validation_f1),"threshold":float(VALIDATION_THRESHOLD),"coverage":float(VALIDATION_COVERAGE)}}
run_metadata={
 "completed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "python":sys.version,"platform":platform.platform(),"dependency_versions":ACTUAL,"gpu_devices":[device.name for device in gpus],
 "dataset_archive_sha256":hashlib.sha256(DATASET_ZIP.read_bytes()).hexdigest(),
 "dataset_source_version":provenance["primary_source"].get("source_version"),
 "dataset_review_status":provenance.get("review_status"),
 "timing_seconds":{
  "source_download":provenance["primary_source"].get("download_time_seconds"),
  "dataset_preparation":provenance["prepared_subset"].get("preparation_seconds"),
  "colab_extract":EXTRACT_SECONDS,"pretrained_model_setup":weight_setup_seconds,
  "frozen_head_training":HEAD_SECONDS,"fine_tuning":FINETUNE_SECONDS,
  "total_training":HEAD_SECONDS+FINETUNE_SECONDS,"evaluation":EVALUATION_SECONDS,
 },
 "limitations":["No signer/session IDs; no unseen-signer claim.","ASL-competent semantic label review pending.","Dataset scores do not measure webcam performance."]
}
for name,data in [("labels.json",labels_export),("preprocessing.json",preprocess_export),("training_config.json",training_config),("training_metrics.json",metrics_export),("run_metadata.json",run_metadata)]:
    (MODEL_DIR/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\\n",encoding="utf-8")
report=[
 "# SignBridge Core training report","",
 f"- Test accuracy: {test_metrics['accuracy']:.4f}",
 f"- Test macro-F1: {test_metrics['macro_f1']:.4f}",
 f"- Validation-only rejection threshold: {VALIDATION_THRESHOLD:.3f}",
 f"- Test accepted coverage: {test_metrics['accepted_coverage']:.3f}",
 f"- Training time: {(HEAD_SECONDS+FINETUNE_SECONDS)/60:.1f} minutes",
 f"- Evaluation time: {EVALUATION_SECONDS:.1f} seconds",
 f"- GPU devices: {gpus if gpus else 'none; CPU run'}",
 f"- Label review: {provenance.get('review_status')}",
 "",
 "Metrics describe only this prepared dataset split. They do not establish semantic ASL correctness, unseen-signer generalization, or webcam performance."
]
(REPORT_DIR/"training_report.md").write_text("\\n".join(report)+"\\n",encoding="utf-8")
reloaded=tf.keras.models.load_model(final_model_path,compile=False)
raw=tf.io.decode_jpeg(tf.io.read_file(val_paths[0]),channels=3)
sample=tf_preprocess_tensor(raw)[tf.newaxis,...]
smoke=reloaded.predict(sample,verbose=0)
assert smoke.shape==(1,20) and np.isfinite(smoke).all()
run_metadata["model_reload_smoke_test"]={"passed":True,"output_shape":list(smoke.shape),"sample_split":"validation"}
(MODEL_DIR/"run_metadata.json").write_text(json.dumps(run_metadata,indent=2,ensure_ascii=False)+"\\n",encoding="utf-8")
print("Saved model:",final_model_path)
print("Reload/inference smoke test passed:",smoke.shape,"| top:",CLASSES[int(smoke[0].argmax())])
"""),
md("""## Optional external holdout

The separate Kaggle ASL Alphabet Test listing currently reports 870 images, 30 per symbol, 200x200, and CC0. Use it only after the downloaded archive's labels and exact/near-duplicate overlap are audited. The notebook skips it unless an external audit JSON says external_eval_ready is true.
"""),
py("""external_zip=DATA_DIR/"external"/"asl-alphabet-test.zip"
external_audit_path=DATA_DIR/"external"/"external_audit.json"
if not external_zip.is_file() or not external_audit_path.is_file():
    print("External evaluation skipped: archive and verified no-overlap audit are required.")
else:
    external_audit=json.loads(external_audit_path.read_text(encoding="utf-8"))
    if external_audit.get("external_eval_ready") is not True:
        print("External evaluation skipped:",external_audit.get("blockers",["audit not ready"]))
    else:
        print("External audit is ready. Review its label and overlap report before enabling a separate evaluation.")
"""),
md("""## Timing and handoff

Run metadata separates preparation, source-download time where it was recorded, model setup, training, and evaluation. Without a GPU, the notebook reports the change and may finish with fewer epochs. It makes no accuracy promise.

Copy best_model.keras plus labels.json, preprocessing.json, and training_config.json from Drive models/ into the local project models/ folder. Then follow the GUI camera checklist in README.md. A trained dataset model and a successful reload smoke test do not count as actual webcam verification.
""")
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "signbridge_core_training.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}
notebook_dir = Path(__file__).resolve().parent / "notebooks"
notebook_dir.mkdir(parents=True, exist_ok=True)
out = notebook_dir / "signbridge_core_training.ipynb"
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {out} ({len(cells)} cells)")
