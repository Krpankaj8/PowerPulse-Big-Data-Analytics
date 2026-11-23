"""Keep regression and classification comparisons on separate metric scales."""


def compare_models(regression_results: dict, classification_results: dict) -> dict:
    """Rank regression by RMSE and classification by high-demand F1 independently."""
    regression = [
        {"model": name, **metrics}
        for name, metrics in regression_results.items()
    ]
    classification = [
        {"model": name, **metrics}
        for name, metrics in classification_results.items()
    ]
    return {
        "regression_by_rmse": sorted(regression, key=lambda result: result["RMSE"]),
        "classification_by_f1_high": sorted(
            classification, key=lambda result: result["f1_high"], reverse=True
        ),
    }
