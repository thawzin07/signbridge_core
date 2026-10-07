# SignBridge public web demo

This folder is a separate Vercel target for the selected `best_model.keras` artifact. It serves a minimalist browser UI, keeps the live preview in the browser, and sends one compressed 320×320 JPEG snapshot per inference request to the Python function. Requests are consent-gated in the interface, limited to 1 MB, processed in memory, and never written to disk by this application. The uploaded-photo path uses the same model endpoint.

The Python function uses TensorFlow CPU and the shared `preprocessing.py` contract. Vercel is configured for Python 3.13 and the Singapore region (`sin1`). A warm function instance caches one model load; each instance has its own model copy. This is a small class-demo deployment, not a production inference service.

## Run locally

Use a Python 3.13 environment and install `requirements.txt`. From this directory, run the Vercel development server with the Vercel CLI (`vercel dev`). The web page is served from the project root and `/api/predict` is the Python function.

## Scope and evidence

- The model classifies 20 static image labels: `A B C D E F G H I K L O P Q R U V W X Y`.
- J and Z, dynamic signs, words, and sentence translation are outside scope.
- The recorded 99.0% accuracy and 0.990 macro-F1 apply only to the prepared CSV test split. They do not establish live webcam accuracy.
- Dataset semantic review by an ASL-competent reviewer is still pending.
- `getUserMedia` requires HTTPS and browser permission; Vercel provides HTTPS.

Before sharing the URL, manually test browser permission, one image upload, one camera session, stopping the camera, stable-vote behavior, and function cold start. The deployment is public and its inference endpoint can be called by anyone who has the URL; do not upload sensitive images.
