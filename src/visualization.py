import numpy as np
import pandas as pd

from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.neighbors import NearestNeighbors

from matplotlib import pyplot as plt
import plotly.express as px
import seaborn as sns


def draw_roc_an_recall_curve(model, X_train, y_train, X_test, y_test):
    # Créer deux sous-graphes côte à côte
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Tracer la courbe ROC sur le premier sous-graphe
    roc_display = RocCurveDisplay.from_estimator(
        model, X_train, y_train, ax=ax1, color="blue", name="Train ROC Curve"
    )
    roc_display = RocCurveDisplay.from_estimator(
        model, X_test, y_test, ax=ax1, color="red", name="Test ROC Curve"
    )
    ax1.set_title("ROC Curve")

    # Tracer la courbe Precision-Recall sur le deuxième sous-graphe
    pr_display = PrecisionRecallDisplay.from_estimator(
        model,
        X_train,
        y_train,
        ax=ax2,
        color="blue",
        name="Train Precision-Recall Curve",
    )
    pr_display = PrecisionRecallDisplay.from_estimator(
        model, X_test, y_test, ax=ax2, color="red", name="Test Precision-Recall Curve"
    )
    ax2.set_title("Precision-Recall Curve")

    # Afficher la figure
    plt.show()


def get_kneighbors_distances(X, n_neighbors=20):
    # Suppose X is already defined
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1).fit(X)
    distances, indices = nn.kneighbors(X)

    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]

    # Créer un DataFrame pour PlotlyXGBClassifier
    df = pd.DataFrame({"index": range(len(distances)), "distance": distances})

    # Créer le graphique avec Plotly Express
    fig = px.line(df, x="index", y="distance", title="K-Nearest Neighbors Distances")

    # Limiter l'axe y entre 0 et 2
    fig.update_yaxes(range=[0, 2])

    # Afficher le graphique
    fig.show()


def plot_highly_correlated(df, threshold=0.8, N=10):
    """
    Displays a heatmap of the most correlated columns according to a given threshold.

    Parameters:
    - df: DataFrame pandas containing the data.
    - threshold: Absolute correlation threshold at which pairs are considered (default 0.8).
    - N: Number of most correlated column pairs to display (default 10).
    """
    # Select only the numeric columns in the dataframe
    df_numeric = df.select_dtypes(include=[np.number])

    # Calculate the correlation matrix for numeric columns
    correlation_matrix = df_numeric.corr()

    # Unstack the correlation matrix to create pairs of columns
    corr_pairs = correlation_matrix.unstack()

    # Remove auto-correlations (correlation of a column with itself)
    corr_pairs = corr_pairs[
        corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)
    ]

    # Filter pairs with correlation greater than the threshold (absolute value)
    highly_correlated_pairs = corr_pairs[abs(corr_pairs) > threshold]

    # Sort the correlation pairs by absolute value and select the top N pairs
    top_corr_pairs = highly_correlated_pairs.abs().sort_values(ascending=False).head(N)

    # Display the top N most correlated pairs
    print(f"Top {N} highly correlated pairs (threshold > {threshold}):")
    print(top_corr_pairs)

    # Extract the unique columns involved in the top N correlated pairs
    top_columns = set()
    for pair in top_corr_pairs.index:
        top_columns.update(pair)

    # Convert top_columns set to a list
    top_columns_list = list(top_columns)

    # Create a subset of the correlation matrix with only the top N correlated columns
    top_corr_matrix = correlation_matrix.loc[top_columns_list, top_columns_list]

    # Change to percentage
    top_corr_matrix = top_corr_matrix * 100

    # Plot the heatmap using Plotly Express
    fig = px.imshow(
        top_corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title=f"Top {N} Correlated Numeric Columns Heatmap",
    )

    fig.update_layout(
        width=None,
        height=800,
        xaxis_title="Columns",
        yaxis_title="Columns",
        autosize=True,
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=45),
    )

    fig.update_traces(
        texttemplate="%{z:.1f}%",
        textfont=dict(size=14),
    )

    fig.show()
