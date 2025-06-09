from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import pandas as pd
import tempfile
import os
from typing import Dict, List

from starlette.middleware.cors import CORSMiddleware
from services.prepare_model import model_predict

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=List[Dict])
async def predict(file: UploadFile = File(...,
                                          media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")):
    try:
        if not file.filename.lower().endswith('.xlsx'):
            raise HTTPException(status_code=400, detail="Требуется файл Excel (.xlsx)")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        result_df = model_predict(tmp_path)

        os.unlink(tmp_path)

        result = result_df.to_dict(orient='records')

        return JSONResponse(
            content=jsonable_encoder(result),
            media_type="application/json; charset=utf-8"
        )

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Файл пуст или имеет неверный формат")
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(
            status_code=500,
            detail=str(e),
            headers={"Content-Type": "application/json; charset=utf-8"}
        )
