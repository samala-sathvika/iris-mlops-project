# Iris MLOps Starter Project

A beginner-friendly end-to-end MLOps example: data validation, model training, MLflow experiment tracking, evaluation, model artifact versioning, FastAPI, pytest, Docker, GitHub, and Render.

## 1. Create and activate a virtual environment (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt:
```bat
.venv\Scripts\activate.bat
```

## 2. Train the model

```powershell
python train.py
```

This downloads/loads scikit-learn's built-in Iris dataset, validates it, trains Logistic Regression, prints metrics, saves `models/iris_model.joblib`, and logs experiment details to `mlruns/`.

## 3. Run the API locally

```powershell
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` and test `POST /predict` with:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

## 4. View MLflow experiments

In a second terminal, activate the same environment and run:
```powershell
mlflow ui --host 127.0.0.1 --port 5000
```
Open `http://127.0.0.1:5000`.

## 5. Run tests

Make sure the model has been trained first:
```powershell
pytest -q
```

## 6. Test Docker locally

Install and start Docker Desktop first. From this project folder:
```powershell
docker build -t iris-mlops .
docker run --rm -p 10000:10000 iris-mlops
```
Open `http://localhost:10000/docs`.

## 7. Push to GitHub

Create an empty GitHub repository (do not add a README if you already have this project README), then run:
```powershell
git init
git add .
git commit -m "Build Iris MLOps project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```
Replace the URL with your actual repository URL. Ensure `models/iris_model.joblib` is included; the model must exist before deploying.

## 8. Deploy on Render using Docker

1. Push the project to GitHub.
2. In Render, choose **New + → Web Service** and connect your repository.
3. Select **Docker** as the runtime/language so Render builds the repository's `Dockerfile`.
4. Keep the root directory blank if the Dockerfile is at the repository root.
5. Create the service. The Dockerfile listens on `0.0.0.0` and uses Render's `PORT` environment variable.
6. When deployment finishes, open your Render service URL, then append `/docs` to test the API.

## Important beginner notes

- Iris is a small educational dataset, so strong test accuracy does not mean the model is ready for safety-critical real-world decisions.
- `models/iris_model.joblib` is a generated artifact. Retrain with `python train.py` after changing the training code.
- The local `mlruns/` folder is for learning. A production team would configure durable/shared experiment tracking storage.
- Never load untrusted `.joblib` files: pickle-based model formats can execute code during loading.
