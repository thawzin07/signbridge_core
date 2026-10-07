# Model artifacts

`best_model.keras`, `labels.json`, `preprocessing.json`, `training_config.json`, `training_metrics.json`, and `run_metadata.json` are the completed Colab export for the 20-class static-letter model. The selected held-out CSV test result is 99.0% accuracy and 0.990 macro-F1 on 1,200 prepared images; this does not establish webcam accuracy or ASL semantic correctness. ASL-competent label review is still pending.

`hand_landmarker.task` is Google's MediaPipe hand-presence model asset used by the local GUI. It only locates a hand; it does not classify the sign. The SignBridge MobileNetV2 remains the letter classifier.
