"""
Makine Öğrenmesi Final — Wine Dataset (load_wine)
================================================
Kapsam:
- Veri: sklearn.datasets.load_wine()
- Split: %70 Train / %10 Validation / %20 Test (Stratified)
- Pipeline (leakage yok): StandardScaler + (KNN / SVM / MLP)
- KNN: Baseline(k=5) + k-F1 grafiği + GridSearch
- SVM: Linear & RBF + GridSearch (C:[0.01,0.1,1,10,100], gamma:['scale','auto',0.01,0.1,1]) + ROC
- MLP: Baseline + RandomizedSearch + loss curve
- Validation tablosu: metrikler + en iyi hiperparametre özeti
- En iyi model seçimi: Validation F1-macro öncelikli (tie-break ROC-AUC)
- Final model: Train+Val ile refit, Test metrikleri + Confusion Matrix + ROC
- Threshold senaryoları: reject option örneği
- KMeans: k=2..10 Elbow+Silhouette + ARI/NMI + 2 özellik scatter + küme profili (>=5 özellik)
- SHAP: KernelExplainer + summary + bar (opsiyonel; shap kurulu olmalı)

Gereksinimler:
    pip install numpy pandas matplotlib scikit-learn shap

Not:
- Bu script grafikleri plt.show() ile açar.
- SHAP bölümü shap yüklü değilse otomatik atlanır.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.cluster import KMeans

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
    adjusted_rand_score, normalized_mutual_info_score,
    classification_report, silhouette_score
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# -------------------------
# Helper functions
# -------------------------
def evaluate_multiclass(model, X_eval, y_eval, average="macro"):
    """Return metrics dict for a multiclass classifier."""
    y_pred = model.predict(X_eval)
    metrics = {
        "accuracy": accuracy_score(y_eval, y_pred),
        "precision_macro": precision_score(y_eval, y_pred, average=average, zero_division=0),
        "recall_macro": recall_score(y_eval, y_pred, average=average, zero_division=0),
        "f1_macro": f1_score(y_eval, y_pred, average=average, zero_division=0),
    }
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_eval)
        y_bin = label_binarize(y_eval, classes=np.unique(y_eval))
        metrics["roc_auc_ovr_macro"] = roc_auc_score(
            y_bin, y_proba, multi_class="ovr", average="macro"
        )
    else:
        metrics["roc_auc_ovr_macro"] = np.nan
    return metrics


def plot_confusion(y_true, y_pred, class_names, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap=None, colorbar=False)
    ax.set_title(title)
    plt.show()


def plot_multiclass_roc(model, X_eval, y_eval, class_names, title="ROC Curves (OvR)"):
    if not hasattr(model, "predict_proba"):
        print("ROC için predict_proba gerekli.")
        return
    y_proba = model.predict_proba(X_eval)
    classes = np.unique(y_eval)
    y_bin = label_binarize(y_eval, classes=classes)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, label=f"{class_names[cls]} (AUC={auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    plt.show()


def summarize_best_params(pipeline_model, model_name):
    """Human-readable best hyperparams summary for the validation table."""
    if model_name == "KNN":
        p = pipeline_model.named_steps["knn"].get_params()
        return f"n_neighbors={p['n_neighbors']}, weights={p['weights']}, p={p['p']}"
    if model_name == "SVM_Linear":
        p = pipeline_model.named_steps["svm"].get_params()
        return f"kernel=linear, C={p['C']}"
    if model_name == "SVM_RBF":
        p = pipeline_model.named_steps["svm"].get_params()
        return f"kernel=rbf, C={p['C']}, gamma={p['gamma']}"
    if model_name == "MLP":
        p = pipeline_model.named_steps["mlp"].get_params()
        return f"hidden={p['hidden_layer_sizes']}, act={p['activation']}, alpha={p['alpha']}, lr={p['learning_rate_init']}"
    return "-"


def predict_with_reject(model, X_eval, threshold=0.7):
    """Reject option: max proba < threshold => Unknown (-1)."""
    proba = model.predict_proba(X_eval)
    pred = proba.argmax(axis=1)
    maxp = proba.max(axis=1)
    pred_reject = pred.copy()
    pred_reject[maxp < threshold] = -1
    return pred_reject


def report_reject(y_true, y_pred_reject):
    unknown_rate = float((y_pred_reject == -1).mean())
    mask = y_pred_reject != -1
    if mask.sum() == 0:
        return {"known_rate": 0.0, "unknown_rate": unknown_rate}
    return {
        "known_rate": float(mask.mean()),
        "unknown_rate": unknown_rate,
        "accuracy_known": accuracy_score(y_true[mask], y_pred_reject[mask]),
        "f1_macro_known": f1_score(y_true[mask], y_pred_reject[mask], average="macro", zero_division=0),
    }


def main():
    # -------------------------
    # 1) Load data
    # -------------------------
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    class_names = wine.target_names

    print("X shape:", X.shape)  # (178, 13)
    print("Num classes:", len(np.unique(y)))
    print("Class distribution:", dict(pd.Series(y).value_counts().sort_index()))

    # -------------------------
    # 2) Split: 70/10/20
    # -------------------------
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    # 0.8 * 0.125 = 0.10
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, random_state=RANDOM_STATE, stratify=y_trainval
    )
    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    # -------------------------
    # 4) KNN
    # -------------------------
    knn_baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ])
    knn_baseline.fit(X_train, y_train)
    print("KNN baseline (val):", evaluate_multiclass(knn_baseline, X_val, y_val))

    # k-F1 graph
    k_list = list(range(1, 32, 2))
    f1_list = []
    for k in k_list:
        tmp = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ])
        tmp.fit(X_train, y_train)
        y_pred = tmp.predict(X_val)
        f1_list.append(f1_score(y_val, y_pred, average="macro"))

    plt.figure(figsize=(7, 4))
    plt.plot(k_list, f1_list, marker="o")
    plt.xlabel("k (n_neighbors)")
    plt.ylabel("Validation F1-macro")
    plt.title("KNN: k - F1 ilişkisi")
    plt.grid(True)
    plt.show()

    # GridSearch for KNN
    knn_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ])
    knn_param_grid = {
        "knn__n_neighbors": list(range(1, 32, 2)),
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }
    knn_gs = GridSearchCV(knn_pipe, knn_param_grid, scoring="f1_macro", cv=5, n_jobs=-1)
    knn_gs.fit(X_train, y_train)
    knn_best = knn_gs.best_estimator_
    print("Best KNN params:", knn_gs.best_params_)
    print("Best KNN CV F1:", knn_gs.best_score_)
    knn_val_metrics = evaluate_multiclass(knn_best, X_val, y_val)

    # -------------------------
    # 5) SVM Linear & RBF
    # -------------------------
    C_grid = [0.01, 0.1, 1, 10, 100]
    gamma_grid = ["scale", "auto", 0.01, 0.1, 1]

    svm_linear = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)),
    ])
    svm_rbf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    ])

    svm_lin_gs = GridSearchCV(svm_linear, {"svm__C": C_grid}, scoring="f1_macro", cv=5, n_jobs=-1)
    svm_rbf_gs = GridSearchCV(
        svm_rbf, {"svm__C": C_grid, "svm__gamma": gamma_grid}, scoring="f1_macro", cv=5, n_jobs=-1
    )
    svm_lin_gs.fit(X_train, y_train)
    svm_rbf_gs.fit(X_train, y_train)

    svm_lin_best = svm_lin_gs.best_estimator_
    svm_rbf_best = svm_rbf_gs.best_estimator_

    print("Best SVM-Linear params:", svm_lin_gs.best_params_, "CV F1:", svm_lin_gs.best_score_)
    print("Best SVM-RBF params:", svm_rbf_gs.best_params_, "CV F1:", svm_rbf_gs.best_score_)

    svm_lin_val_metrics = evaluate_multiclass(svm_lin_best, X_val, y_val)
    svm_rbf_val_metrics = evaluate_multiclass(svm_rbf_best, X_val, y_val)

    plot_multiclass_roc(svm_lin_best, X_val, y_val, class_names, title="SVM Linear — ROC (Validation, OvR)")
    plot_multiclass_roc(svm_rbf_best, X_val, y_val, class_names, title="SVM RBF — ROC (Validation, OvR)")

    # -------------------------
    # 6) MLP
    # -------------------------
    mlp_baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(max_iter=2000, random_state=RANDOM_STATE, early_stopping=True)),
    ])
    mlp_baseline.fit(X_train, y_train)
    print("MLP baseline (val):", evaluate_multiclass(mlp_baseline, X_val, y_val))

    mlp_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(max_iter=2000, random_state=RANDOM_STATE, early_stopping=True)),
    ])

    mlp_param_dist = {
        "mlp__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "mlp__learning_rate_init": [0.001, 0.01],
    }

    mlp_rs = RandomizedSearchCV(
        mlp_pipe, mlp_param_dist, n_iter=20, scoring="f1_macro", cv=5,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    mlp_rs.fit(X_train, y_train)
    mlp_best = mlp_rs.best_estimator_
    print("Best MLP params:", mlp_rs.best_params_)
    print("Best MLP CV F1:", mlp_rs.best_score_)
    mlp_val_metrics = evaluate_multiclass(mlp_best, X_val, y_val)

    # Loss curve
    best_mlp_model = mlp_best.named_steps["mlp"]
    plt.figure(figsize=(7, 4))
    plt.plot(best_mlp_model.loss_curve_)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("MLP Loss Curve (Best Model)")
    plt.grid(True)
    plt.show()

    # -------------------------
    # 7) Validation table
    # -------------------------
    models = {
        "KNN": knn_best,
        "SVM_Linear": svm_lin_best,
        "SVM_RBF": svm_rbf_best,
        "MLP": mlp_best,
    }
    val_metrics_map = {
        "KNN": knn_val_metrics,
        "SVM_Linear": svm_lin_val_metrics,
        "SVM_RBF": svm_rbf_val_metrics,
        "MLP": mlp_val_metrics,
    }

    rows = []
    for name, mdl in models.items():
        row = {"model": name, "best_params": summarize_best_params(mdl, name)}
        row.update(val_metrics_map[name])
        rows.append(row)

    val_table = pd.DataFrame(rows).set_index("model")
    print("\nValidation Comparison Table:\n")
    print(val_table)

    # -------------------------
    # 8) Select best model
    # -------------------------
    val_table_sorted = val_table.sort_values(by=["f1_macro", "roc_auc_ovr_macro"], ascending=False)
    best_name = val_table_sorted.index[0]
    best_model = models[best_name]
    print("\nSelected best model:", best_name)

    # -------------------------
    # 9) Refit with Train+Val, evaluate on Test
    # -------------------------
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    final_model = best_model
    final_model.fit(X_train_full, y_train_full)

    test_metrics = evaluate_multiclass(final_model, X_test, y_test)
    print("\nTest metrics:\n", test_metrics)

    y_test_pred = final_model.predict(X_test)
    plot_confusion(y_test, y_test_pred, class_names, title=f"{best_name} — Confusion Matrix (Test)")
    print("\nClassification report:\n")
    print(classification_report(y_test, y_test_pred, target_names=class_names))

    plot_multiclass_roc(final_model, X_test, y_test, class_names, title=f"{best_name} — ROC (Test, OvR)")

    # Threshold scenarios (reject option)
    print("\nThreshold (reject option) scenarios:")
    for t in [0.8, 0.6]:
        pred_r = predict_with_reject(final_model, X_test, threshold=t)
        print(f"  threshold={t} ->", report_reject(y_test, pred_r))

    # -------------------------
    # 11) KMeans
    # -------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ks = range(2, 11)
    inertias, sil_scores = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    plt.figure(figsize=(7, 4))
    plt.plot(list(ks), inertias, marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("KMeans Elbow (Inertia vs k)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(list(ks), sil_scores, marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.title("KMeans Silhouette vs k")
    plt.grid(True)
    plt.show()

    best_k = list(ks)[int(np.argmax(sil_scores))]
    print("\nChosen k (by max silhouette):", best_k)

    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    ari = adjusted_rand_score(y, clusters)
    nmi = normalized_mutual_info_score(y, clusters)
    print("ARI:", ari)
    print("NMI:", nmi)

    # Scatter with 2 features (no dimensionality reduction)
    f1_name, f2_name = "alcohol", "color_intensity"
    i1, i2 = feature_names.index(f1_name), feature_names.index(f2_name)

    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, i1], X[:, i2], c=clusters)
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.title(f"KMeans Clusters (k={best_k}) — {f1_name} vs {f2_name}")
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, i1], X[:, i2], c=y)
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.title(f"True Classes — {f1_name} vs {f2_name}")
    plt.show()

    # Cluster profile (>=5 features)
    profile_features = ["alcohol", "malic_acid", "ash", "color_intensity", "proline"]
    idxs = [feature_names.index(f) for f in profile_features]
    profile_df = pd.DataFrame(X[:, idxs], columns=profile_features)
    profile_df["cluster"] = clusters
    cluster_profile = profile_df.groupby("cluster").mean().round(3)
    print("\nCluster profile (mean of 5 features):\n")
    print(cluster_profile)

    # -------------------------
    # 12) SHAP (optional)
    # -------------------------
    try:
        import shap  # type: ignore

        print("\nSHAP is available. Running SHAP KernelExplainer (may take a bit)...")
        X_bg = shap.sample(X_train_full, 100, random_state=RANDOM_STATE)
        X_explain = shap.sample(X_test, 30, random_state=RANDOM_STATE)

        explainer = shap.KernelExplainer(final_model.predict_proba, X_bg)
        shap_values = explainer.shap_values(X_explain, nsamples=200)

        target_class = 0
        shap.summary_plot(
            shap_values[target_class] if isinstance(shap_values, list) else shap_values,
            X_explain,
            feature_names=feature_names,
            show=True
        )

        shap.summary_plot(
            shap_values[target_class] if isinstance(shap_values, list) else shap_values,
            X_explain,
            feature_names=feature_names,
            plot_type="bar",
            show=True
        )
    except Exception as e:
        print("\nSHAP bölümü atlandı. (shap kurulu değil ya da çalıştırılamadı)")
        print("Detay:", repr(e))

    print("\nDONE.")


if __name__ == "__main__":
    main()
