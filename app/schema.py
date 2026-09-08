"""
schema.py
=========
Pydantic models للـ API - المدخلات هنا هي البيانات "الخام" لكل أوردر
(قبل الـ feature engineering)، بنفس الحقول اللي استُخدمت وقت التدريب
في features.py::apply_feature_engineering() + config.FEATURE_COLS.
"""

from typing import List
from pydantic import BaseModel, Field


class OrderFeatures(BaseModel):
    """
    البيانات الخام المطلوبة لكل أوردر.
    ملحوظة: الأعمدة المشتقة (purchase_year, is_same_state, zip_prefix_diff, ...)
    بيتم حسابها تلقائيًا جوه preprocess_data() ومحتاجش تُرسل من العميل.
    """

    # وقت الشراء - لازم يكون قابل للتحويل لـ datetime (ISO format مثلاً)
    order_purchase_timestamp: str = Field(..., examples=["2018-05-14 10:23:00"])

    # جغرافيا
    customer_state: str = Field(..., examples=["SP"])
    seller_state: str = Field(..., examples=["RJ"])
    customer_zip_code_prefix: float = Field(..., examples=[14409.0])
    seller_zip_code_prefix: float = Field(..., examples=[27277.0])

    # أوردر / منتجات
    total_items: int = Field(..., examples=[2])
    num_sellers: int = Field(..., examples=[1])
    total_item_price: float = Field(..., examples=[150.0])
    total_freight: float = Field(..., examples=[25.5])
    avg_item_price: float = Field(..., examples=[75.0])
    num_categories: int = Field(..., examples=[1])
    avg_product_weight_g: float = Field(..., examples=[800.0])

    # مدفوعات
    num_payment_installments_types: int = Field(..., examples=[1])
    max_installments: int = Field(..., examples=[3])
    total_payment_value: float = Field(..., examples=[175.5])
    payment_types_used: str = Field(..., examples=["credit_card"])


class PredictionResult(BaseModel):
    prediction: int
    probability: float
    model_version: str


class BatchPredictionRequest(BaseModel):
    orders: List[OrderFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    total_count: int
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    source: str
