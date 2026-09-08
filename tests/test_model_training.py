import numpy as np
from scipy.sparse import csr_matrix

import src.task3.config as config

# افترض أن اسم ملفك الأصلي هو model_training.py (قم بتعديله حسب اسم ملفك)
from src.task3.models import (
    build_logistic_regression,
    build_random_forest,
    calculate_scale_pos_weight,
    evaluate_model,
    save_model,
)


def test_calculate_scale_pos_weight():
    """اختبار حساب وزن الفئات غير المتوازنة بدقة."""
    # 3 حالات سلبية و 2 حالة إيجابية (الناتج المتوقع: 3 / 2 = 1.5)
    y_train = np.array([0, 0, 0, 1, 1])
    weight = calculate_scale_pos_weight(y_train)
    assert weight == 1.5


def test_build_models():
    """اختبار إنشاء نماذج الانحدار اللوجستي والغابة العشوائية."""
    lr = build_logistic_regression(random_state=42)
    rf = build_random_forest(n_estimators=10, random_state=42, n_jobs=1)

    assert lr is not None
    assert rf is not None


def test_evaluate_model():
    """اختبار دالة التقييم واستخراج مقاييس الأداء."""
    # مصفوفة بيانات خفيفة للاختبار
    X_val = csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    y_val = np.array([0, 1])

    lr = build_logistic_regression(random_state=42)
    lr.fit(X_val, y_val)

    metrics = evaluate_model("LogisticRegression", lr, X_val, y_val)

    # التأكد من وجود المقاييس الأساسية وقيمتها ضمن النطاق الصحيح
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_save_model(tmp_path, monkeypatch):
    """اختبار حفظ ملف النموذج بنجاح في مسار مؤقت."""
    model_path = tmp_path / "best_model.pkl"
    monkeypatch.setattr(config, "BEST_MODEL_PATH", model_path)

    lr = build_logistic_regression()
    save_model(lr, filepath=model_path)

    assert model_path.exists()
