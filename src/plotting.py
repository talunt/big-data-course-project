"""Figure defaults with a readable English fallback on systems without CJK fonts."""

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

EN_MODEL_NAMES = {
    "Mean": "Mean",
    "OLS": "Linear regression",
    "Ridge": "Ridge",
    "Lasso": "Lasso",
    "ElasticNet": "Elastic net",
    "PolyRidge": "Ridge with added weather/calendar terms",
    "DecisionTree": "Decision tree",
    "RandomForest": "Random forest",
    "GBT": "Gradient-boosted trees",
    "KNN": "K-nearest neighbors",
}


def configure():
    available = {item.name for item in font_manager.fontManager.ttflist}
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "PingFang SC",
    ]
    chosen = next((name for name in candidates if name in available), None)
    plt.rcParams.update(
        {
            "font.family": chosen or "DejaVu Sans",
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return chosen is not None


def save_figure(folder, name, dpi=180):
    folder.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(folder / name, dpi=dpi, bbox_inches="tight")
    plt.close()
