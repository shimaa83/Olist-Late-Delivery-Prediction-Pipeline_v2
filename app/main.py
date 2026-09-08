import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import onnxruntime as ort
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack, issparse
from fastapi import FastAPI, HTTPException, status

import src.task3.config as config
from app.schema import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    OrderFeatures,
    PredictionResult,
)

# إعداد logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("olist_api")
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))
MODEL_PATH = Path(config.BEST_MODEL_PATH)
ARTIFACTS_DIR = Path(config.ARTIFACTS_DIR)

# قاموس لحفظ الموديل والـ preprocessors
model_assets: Dict[str, Any] = {}


# ==================================================
# تحميل الموديل + كل الـ preprocessors
# ==================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """تحميل ONNX model وكل الـ preprocessors عند بدء السيرفر"""
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"الموديل غير موجود: {MODEL_PATH}")

        logger.info(f"⏳ جاري تحميل ONNX من: {MODEL_PATH}")

        session = ort.InferenceSession(
            str(MODEL_PATH), providers=["CPUExecutionProvider"]
        )

        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]

        logger.info(f"✅ مدخلات ONNX: {input_names}")
        logger.info(f"✅ مخرجات ONNX: {output_names}")

        # 🔥 تحميل كل الـ preprocessors بنفس الترتيب المستخدم في features.py
        cat_imputer = joblib.load(config.CAT_IMPUTER_PATH)
        cat_encoder = joblib.load(config.CAT_ENCODER_PATH)
        num_imputer = joblib.load(config.NUM_IMPUTER_PATH)
        num_scaler = joblib.load(config.NUM_SCALER_PATH)

        categorical_features = joblib.load(config.CATEGORICAL_FEATURES_PATH)
        numerical_features = joblib.load(config.NUMERICAL_FEATURES_PATH)
        final_feature_names = joblib.load(config.FINAL_FEATURE_NAMES_PATH)

        logger.info(f"✅ الأعمدة categorical: {categorical_features}")
        logger.info(f"✅ الأعمدة numerical: {numerical_features}")
        logger.info(
            f"✅ إجمالي الأعمدة النهائية بعد الـ encoding: {len(final_feature_names)}"
        )

        model_assets["cat_imputer"] = cat_imputer
        model_assets["cat_encoder"] = cat_encoder
        model_assets["num_imputer"] = num_imputer
        model_assets["num_scaler"] = num_scaler
        model_assets["categorical_features"] = list(categorical_features)
        model_assets["numerical_features"] = list(numerical_features)
        model_assets["final_feature_names"] = list(final_feature_names)

        model_assets["session"] = session
        model_assets["input_names"] = input_names
        model_assets["output_names"] = output_names
        model_assets["model_version"] = "v1.0.0-onnx"

        logger.info("✅ تم تحميل النموذج وكل الـ preprocessors بنجاح!")

    except Exception as e:
        logger.error(f"❌ فشل تحميل النموذج أو الـ preprocessors: {e}", exc_info=True)
        model_assets["session"] = None

    yield

    logger.info("⏹️ إيقاف السيرفر")
    model_assets.clear()


app = FastAPI(
    title="Olist Order Delay Prediction API",
    description="خدمة التنبؤ بتأخير الشحنات",
    version="1.0.0",
    lifespan=lifespan,
)


# ==================================================
# Feature Engineering - نفس منطق features.py::apply_feature_engineering
# ==================================================
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # تحويل وقت الشراء لـ datetime
    df[config.TIME_SORT_COL] = pd.to_datetime(df[config.TIME_SORT_COL], errors="coerce")

    if df[config.TIME_SORT_COL].isna().any():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"قيمة غير صالحة في {config.TIME_SORT_COL} - تأكد من صيغة التاريخ",
        )

    # log1p على الأعمدة المالية (بنفس ترتيب وطريقة التدريب)
    for col in config.LOG_TRANSFORM_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # تطابق الولاية
    df["is_same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)

    # الفرق في الرمز البريدي
    df["zip_prefix_diff"] = np.abs(
        df["customer_zip_code_prefix"] - df["seller_zip_code_prefix"]
    )

    # ميزات التاريخ
    df["purchase_year"] = df[config.TIME_SORT_COL].dt.year
    df["purchase_month"] = df[config.TIME_SORT_COL].dt.month
    df["purchase_dayofweek"] = df[config.TIME_SORT_COL].dt.dayofweek
    df["is_weekend"] = df["purchase_dayofweek"].isin([5, 6]).astype(int)

    return df


# ==================================================
# دالة معالجة البيانات الكاملة (Feature Eng + Impute + Encode + Scale + Stack)
# ==================================================
def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    try:
        logger.info(f"📥 البيانات الخام المدخلة: {df.columns.tolist()}")

        cat_imputer = model_assets.get("cat_imputer")
        cat_encoder = model_assets.get("cat_encoder")
        num_imputer = model_assets.get("num_imputer")
        num_scaler = model_assets.get("num_scaler")
        categorical_features = model_assets.get("categorical_features", [])
        numerical_features = model_assets.get("numerical_features", [])

        if not all([cat_imputer, cat_encoder, num_imputer, num_scaler]):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="الـ preprocessors غير محمّلة على السيرفر",
            )

        # ✅ خطوة 1: Feature Engineering (نفس اللي حصل وقت التدريب)
        df = apply_feature_engineering(df)

        # ✅ خطوة 2: تأكد من وجود كل أعمدة FEATURE_COLS المطلوبة
        missing = [c for c in config.FEATURE_COLS if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"أعمدة ناقصة بعد الـ feature engineering: {missing}",
            )

        df = df[config.FEATURE_COLS].copy()

        # ✅ خطوة 3: Categorical -> Impute -> Encode (بنفس ترتيب categorical_features)
        X_cat = cat_imputer.transform(df[categorical_features])
        X_cat = cat_encoder.transform(X_cat)  # sparse matrix

        # ✅ خطوة 4: Numerical -> Impute -> Scale (بنفس ترتيب numerical_features)
        X_num = num_imputer.transform(df[numerical_features])
        X_num = num_scaler.transform(X_num)

        # ✅ خطوة 5: Stack بنفس ترتيب التدريب: numerical أولاً ثم categorical
        X_final = hstack([X_num, X_cat])
        if issparse(X_final):
            X_final = X_final.toarray()

        result = np.asarray(X_final).astype("float32")
        logger.info(f"✅ شكل البيانات النهائي: {result.shape}, dtype: {result.dtype}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ خطأ في معالجة البيانات: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"خطأ في معالجة البيانات: {str(e)}",
        )


# ==================================================
# دالة التوقع الرئيسية
# ==================================================
def predict(df: pd.DataFrame) -> List[Dict[str, Any]]:
    session = model_assets.get("session")

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="النموذج غير محمل",
        )

    try:
        processed_data = preprocess_data(df)

        input_names = model_assets.get("input_names", [])
        logger.info(f"📥 أسماء المدخلات المتوقعة: {input_names}")
        logger.info(f"📊 شكل البيانات: {processed_data.shape}")

        if len(input_names) == 1:
            input_name = input_names[0]
            input_data = {input_name: processed_data}
        else:
            logger.error(f"❌ inputs متعددة: {input_names}")
            raise ValueError(f"Unexpected multiple inputs: {input_names}")

        output_names = model_assets.get("output_names", [])
        predictions_raw = session.run(output_names, input_data)

        logger.info(f"✅ النتايج: {[r.shape for r in predictions_raw]}")

        predictions = predictions_raw[0]

        if len(predictions.shape) == 1:
            probabilities = predictions.astype(float)
        else:
            probabilities = predictions[:, 1]

        model_version = model_assets.get("model_version", "unknown")

        results = []
        for i, pred in enumerate(predictions):
            prob = (
                float(probabilities[i])
                if hasattr(probabilities, "__len__")
                else float(probabilities)
            )
            results.append(
                {
                    "prediction": int(pred),
                    "probability": round(prob, 4),
                    "model_version": model_version,
                }
            )

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ خطأ في التوقع: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"خطأ في التوقع: {str(e)}",
        )


# ==================================================
# Endpoints
# ==================================================


@app.get("/health", response_model=HealthResponse, tags=["Observability"])
async def health_check():
    is_loaded = model_assets.get("session") is not None
    return HealthResponse(
        status="healthy" if is_loaded else "unhealthy",
        model_loaded=is_loaded,
    )


@app.get("/info", response_model=ModelInfoResponse, tags=["Observability"])
async def model_info():
    return ModelInfoResponse(
        model_name="Olist_Delay_Model",
        model_version=model_assets.get("model_version", "unknown"),
        source="ONNX Model",
    )


@app.get("/features", tags=["Observability"])
async def required_features():
    """بيرجع الأعمدة الخام المطلوبة من العميل + الأعمدة النهائية بعد المعالجة"""
    return {
        "raw_categorical_features": model_assets.get("categorical_features", []),
        "raw_numerical_features": model_assets.get("numerical_features", []),
        "feature_cols_order": config.FEATURE_COLS,
        "final_features_count_after_encoding": len(
            model_assets.get("final_feature_names", [])
        ),
    }


@app.post("/predict", response_model=PredictionResult, tags=["Inference"])
async def predict_single(order: OrderFeatures):
    """توقع لطلب واحد - بياخد البيانات الخام ويعمل feature engineering كامل"""
    df = pd.DataFrame([order.model_dump()])
    results = predict(df)
    return results[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(batch: BatchPredictionRequest):
    """توقع لعدة طلبات"""
    if not batch.orders:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="قائمة الطلبات فارغة",
        )

    df = pd.DataFrame([order.model_dump() for order in batch.orders])
    results = predict(df)

    return BatchPredictionResponse(
        predictions=[PredictionResult(**r) for r in results],
        total_count=len(results),
        model_version=model_assets.get("model_version", "unknown"),
    )


# تشغيل: uvicorn app.main:app --reload
