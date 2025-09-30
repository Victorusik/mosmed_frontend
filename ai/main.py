from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import zipfile
import time

from model_utils import init_model, process_dicom_study

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="CT Pathology API")

@app.on_event("startup")
def startup_event():
    init_model(model_path=str(BASE_DIR / "best_model.pth"))

@app.post("/predict")
async def predict_pathology(file: UploadFile = File(...)):
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    filename = file.filename
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Upload a .zip archive")

    t0 = time.time()
    tmp_folder = OUTPUT_DIR / (filename.replace(".zip", "") + "_" + str(int(t0)))
    tmp_folder.mkdir(parents=True, exist_ok=True)
    archive_path = tmp_folder / filename

    with open(archive_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        with zipfile.ZipFile(archive_path, 'r') as z:
            z.extractall(tmp_folder)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Bad zip archive")

    try:
        df = process_dicom_study(str(tmp_folder), save_images=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    if "time_of_processing" not in df.columns:
        df["time_of_processing"] = round(time.time() - t0, 3)

    excel_path = tmp_folder / (filename.replace(".zip", "") + "_report.xlsx")
    df.to_excel(excel_path, index=False)

    return FileResponse(str(excel_path), filename=excel_path.name, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
