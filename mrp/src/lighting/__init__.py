from .classification_model import ClassificationModel
from .classification_model_mg import ClassificationModel_mg

LIGHTNING_MODULES = {
    "classification": ClassificationModel, #mri
    "classification_mg": ClassificationModel_mg,
}
