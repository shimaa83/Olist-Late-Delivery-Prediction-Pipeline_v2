from pathlib import Path
import joblib
import numpy as np
import pytest
from xgboost import XGBClassifier

import src.task3.config as config
from src.task3.export_onnx import convert_best_model_to_onnx


@pytest.fixture
def mock_onnx_env(tmp_path, monkeypatch):
    """إعداد بيئة وهمية لاختبار تحويل النموذج إلى ONNX."""
    model_path = tmp_path / "best_model.pkl"
    monkeypatch.setattr(config, "BEST_MODEL_PATH", model_path)

    # إنشاء نموذج XGBoost حقيقي ومصغر لضمان نجاح عملية التحويل
    model = XGBClassifier(eval_metric="logloss", n_estimators=2)
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    model.fit(X, y)

    # حفظ النموذج في المسار المؤقت
    joblib.dump(model, model_path)
    return model_path


def test_convert_model_not_found(tmp_path, monkeypatch):
    """اختبار التعامل بشكل صحيح عندما يكون ملف النموذج غير موجود."""
    missing_path = tmp_path / "non_existent.pkl"
    monkeypatch.setattr(config, "BEST_MODEL_PATH", missing_path)

    result = convert_best_model_to_onnx()
    assert result is None


def test_convert_best_model_to_onnx_success(mock_onnx_env):
    """اختبار التحويل الناجح والتأكد من إنشاء ملف ONNX بنجاح."""
    onnx_path = convert_best_model_to_onnx()

    assert onnx_path is not None
    assert isinstance(onnx_path, Path)
    assert onnx_path.exists()
    assert onnx_path.suffix == ".onnx"
