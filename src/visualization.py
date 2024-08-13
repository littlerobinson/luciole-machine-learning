from matplotlib import pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def draw_roc_an_recall_curve(model, X_train, y_train, X_test, y_test):
    # Créer deux sous-graphes côte à côte
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Tracer la courbe ROC sur le premier sous-graphe
    roc_display = RocCurveDisplay.from_estimator(
        model, X_train, y_train, ax=ax1, color='blue', name='Train ROC Curve'
    )
    roc_display = RocCurveDisplay.from_estimator(
        model, X_test, y_test, ax=ax1, color='red', name='Test ROC Curve'
    )
    ax1.set_title('ROC Curve')

    # Tracer la courbe Precision-Recall sur le deuxième sous-graphe
    pr_display = PrecisionRecallDisplay.from_estimator(
        model, X_train, y_train, ax=ax2, color='blue', name='Train Precision-Recall Curve'
    )
    pr_display = PrecisionRecallDisplay.from_estimator(
        model, X_test, y_test, ax=ax2, color='red', name='Test Precision-Recall Curve'
    )
    ax2.set_title('Precision-Recall Curve')

    # Afficher la figure
    plt.show()