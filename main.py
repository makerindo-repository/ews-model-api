from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from datetime import date
import joblib
import pandas as pd

try:
    model = joblib.load("prophet_model_v1.pkl")
except FileNotFoundError:
    raise RuntimeError("Model file not found or not in .pkl format")


app = FastAPI(title="EWS Model Flood Prediction V1", version="1.0")


class PredictRequest(BaseModel):
    start_date: date = Field(..., description="Tanggal awal prediksi (YYYY-MM-DD)")
    days: int = Field(..., gt=0, le=30, description="Jumlah hari prediksi (1-30)")

    @field_validator("start_date")
    def validate_start_date(cls, v: date):
        if v < date.today():
            raise ValueError("start_date tidak boleh lebih kecil dari hari ini")
        return v


class PredictResponse(BaseModel):
    date: date
    yhat: float
    yhat_lower: float
    yhat_upper: float
    label: str


@app.post("/predict", response_model=list[PredictResponse])
def predict(req: PredictRequest):
    try:
        future = pd.date_range(start=req.start_date, periods=req.days, freq="D")
        future_df = pd.DataFrame({"ds": future})

        forecast = model.predict(future_df)

        forecast["label"] = forecast["yhat"].apply(
            lambda x: "Banjir" if x > 500 else "Tidak Banjir"
        )

        results = [
            PredictResponse(
                date=row.ds.date(),
                yhat=row.yhat,
                yhat_lower=row.yhat_lower,
                yhat_upper=row.yhat_upper,
                label=row.label,
            )
            for row in forecast.itertuples()
        ]

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
