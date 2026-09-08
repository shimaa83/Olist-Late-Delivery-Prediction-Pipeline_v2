import logging
from pathlib import Path

import joblib
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

import src.task3.config as config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("export_onnx")


def convert_best_model_to_onnx():
    """تحويل نموذج XGBoost إلى صيغة ONNX عبر تجميع الخصائص في Tensor موحد."""
    model_path = Path(config.BEST_MODEL_PATH)
    if not model_path.exists():
        logger.error(f"لم يتم العثور على النموذج الممرن في المسار: {model_path}")
        return

    logger.info(f"جاري تحميل النموذج من {model_path}...")
    model = joblib.load(model_path)

    # معرفة عدد الخصائص التي يتوقعها النموذج تلقائيًا
    if hasattr(model, "n_features_in_"):
        num_features = model.n_features_in_
    else:
        num_features = 7  # العدد الافتراضي للخصائص المستخدمة

    # XGBoost يتوقع Vector موحد للمدخلات العددية
    initial_type = [("float_input", FloatTensorType([None, num_features]))]

    logger.info(
        f"جاري تحويل نموذج XGBoost (عدد الخصائص: {num_features}) إلى صيغة ONNX..."
    )
    try:
        onnx_model = convert_xgboost(
            model,
            initial_types=initial_type,
            target_opset=15,
        )

        onnx_output_path = model_path.with_suffix(".onnx")
        with open(onnx_output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        logger.info(f"تم حفظ نموذج ONNX بنجاح في: {onnx_output_path}")
        return onnx_output_path
    except Exception as e:
        logger.error(f"فشل تحويل النموذج إلى ONNX: {str(e)}")
        raise e


if __name__ == "__main__":
    convert_best_model_to_onnx()
