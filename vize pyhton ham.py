
# Gerekli kütüphanelerin yüklenmesi
!pip install shap xgboost

# Temel Kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Scikit-Learn Modülleri
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Modeller
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

# Metrikler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve, auc)

# XAI (Açıklanabilirlik)
import shap

# Ayarlar
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
print("Kütüphaneler başarıyla yüklendi.")

# 1.1 Veri Setini Yükleme
data = load_wine()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

# 1.2 DataFrame Oluşturma
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# İlk 5 satırın gösterimi
print("Veri Setinin İlk 5 Satırı:")
display(df.head())

# 2.1 Eksik Değer Analizi
missing_values = df.isnull().sum()
print("\n--- Eksik Değer Tablosu ---")
print(missing_values[missing_values > 0])

if missing_values.sum() == 0:
    print("Veri setinde eksik değer bulunmamaktadır.")
else:
    df.fillna(df.mean(), inplace=True)
    print("Eksik değerler ortalama ile dolduruldu.")

# 2.2 Aykırı Değer (Outlier) Analizi - IQR Yöntemi
print("\n--- Aykırı Değer Analizi (IQR) ---")
outlier_report = []

# Hedef değişken hariç özelliklerde döngü
for col in feature_names:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    num_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
    if num_outliers > 0:
        outlier_report.append([col, num_outliers])

outlier_df = pd.DataFrame(outlier_report, columns=["Özellik", "Aykırı Değer Sayısı"])
display(outlier_df)

# 2.3 Veri Tipleri
print("\n--- Veri Tipleri ve Dağılım ---")
print(df.dtypes)
print(f"\nSınıf Dağılımı:\n{df['target'].value_counts()}")

# 3.1 İstatistiksel Özellikler
print("\n--- Temel İstatistikler ---")
display(df.describe().T[['mean', '50%', 'min', 'max', 'std']]) # 50% Medyandır

# 3.2 Korelasyon Matrisi
plt.figure(figsize=(12, 10))
corr = df[feature_names].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title("Özellikler Arası Pearson Korelasyon Matrisi")
plt.show()

# En yüksek korelasyonu bulma
upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
max_corr = upper_tri.unstack().sort_values(ascending=False).head(3)
print(f"\nEn Yüksek Pozitif Korelasyonlu 3 Çift:\n{max_corr}")

# 3.3 Boxplot Analizi
plt.figure(figsize=(20, 8))
df_melted = pd.melt(df, id_vars='target', value_vars=feature_names)
sns.boxplot(x='variable', y='value', data=df_melted)
plt.xticks(rotation=45)
plt.title("Tüm Özelliklerin Boxplot Analizi (Ölçeklendirme Öncesi)")
plt.show()

# StandardScaler kullanımı (Ortalama=0, Varyans=1 yapar)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Kontrol amaçlı DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
print("Veri başarıyla ölçeklendirildi.")
display(X_scaled_df.head(3))

# Adım 1: Önce Test (%20) ve Geriye Kalan (%80) olarak ayır
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

# Adım 2: Geriye Kalan veriyi (%80), Train (%70) ve Validation (%10) oranlarına ayır.
# Validation tüm verinin %10'u olmalı. Elimizde %80 var.
# 10 / 80 = 0.125 oranıyla bölersek orijinal verinin %10'unu elde ederiz.
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
)

print(f"Training Set : {X_train.shape} (%{len(X_train)/len(X)*100:.1f})")
print(f"Validation Set: {X_val.shape}  (%{len(X_val)/len(X)*100:.1f})")
print(f"Test Set      : {X_test.shape}  (%{len(X_test)/len(X)*100:.1f})")

# PCA Eğitimi
pca = PCA()
pca.fit(X_train)

# Explained Variance Ratio
exp_var = pca.explained_variance_ratio_
mean_var = np.mean(exp_var)
n_components_pca = np.sum(exp_var > mean_var)

print(f"Ortalama varyanstan ({mean_var:.4f}) büyük olan bileşen sayısı: {n_components_pca}")

# PCA Explained Variance Grafiği
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(exp_var)+1), exp_var, alpha=0.7, align='center', label='Bireysel varyans')
plt.step(range(1, len(exp_var)+1), np.cumsum(exp_var), where='mid', label='Kümülatif varyans', color='red')
plt.axhline(y=mean_var, color='green', linestyle='--', label='Ortalama Varyans')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.legend(loc='best')
plt.title("PCA Variance Grafiği")
plt.show()

# Seçilen 2 bileşen ile görselleştirme
pca_2d = PCA(n_components=2)
X_train_pca = pca_2d.fit_transform(X_train)
X_val_pca = pca_2d.transform(X_val)
X_test_pca = pca_2d.transform(X_test)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_train_pca[:,0], y=X_train_pca[:,1], hue=y_train, palette='viridis', style=y_train, s=100)
plt.title("PCA ile İndirgenmiş Veri (İlk 2 Bileşen)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# Not: LDA bileşen sayısı sınıf sayısının bir eksiğinden fazla olamaz (C-1).
# Wine veri seti 3 sınıflı olduğu için max bileşen sayısı 2'dir. Soru n=3 istese de matematiksel sınır n=2'dir.
n_lda_components = min(len(np.unique(y)) - 1, 3)

lda = LDA(n_components=n_lda_components)
X_train_lda = lda.fit_transform(X_train, y_train)
X_val_lda = lda.transform(X_val)
X_test_lda = lda.transform(X_test)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_train_lda[:,0], y=X_train_lda[:,1], hue=y_train, palette='viridis', style=y_train, s=100)
plt.title("LDA ile İndirgenmiş Veri")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.show()

# Modeller Sözlüğü
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42),
    "Naive Bayes": GaussianNB()
}

# Veri Temsilleri
data_representations = {
    "Ham Veri": (X_train, X_val),
    "PCA Verisi": (X_train_pca, X_val_pca),
    "LDA Verisi": (X_train_lda, X_val_lda)
}

results = []

# Döngü ile eğitim ve test
for data_name, (X_tr, X_v) in data_representations.items():
    for model_name, model in models.items():
        # Eğitim
        model.fit(X_tr, y_train)

        # Tahmin
        y_pred = model.predict(X_v)
        # Prob (AUC için) - Naive Bayes ve SVM bazen sorun çıkarabilir, kontrol ediyoruz
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_v)
            auc_val = roc_auc_score(y_val, y_prob, multi_class='ovr')
        else:
            auc_val = 0 # Prob yoksa 0

        # Metrikler
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        prec = precision_score(y_val, y_pred, average='weighted')
        rec = recall_score(y_val, y_pred, average='weighted')

        results.append({
            "Veri Tipi": data_name,
            "Model": model_name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc_val
        })

# Sonuçları Tabloya Dökme
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="Accuracy", ascending=False)

print("--- Validation Performans Tablosu ---")
display(df_results)

# En iyi modelin seçimi (Otomatik)
best_row = df_results.iloc[0]
best_model_name = best_row["Model"]
best_data_type = best_row["Veri Tipi"]

print(f"\nSeçilen En İyi Model: {best_model_name} ({best_data_type})")

# Modeli ve Veriyi Hazırlama
model = models[best_model_name]

if best_data_type == "Ham Veri":
    X_test_final = X_test
    X_train_final = X_train
elif best_data_type == "PCA Verisi":
    X_test_final = X_test_pca
    X_train_final = X_train_pca
else: # LDA
    X_test_final = X_test_lda
    X_train_final = X_train_lda

# Modeli train setine tekrar fit ediyoruz (emin olmak için)
model.fit(X_train_final, y_train)

# 9.1 Test Metrikleri
y_test_pred = model.predict(X_test_final)
y_test_prob = model.predict_proba(X_test_final)

print("\n--- TEST SETİ PERFORMANSI ---")
print(f"Accuracy : {accuracy_score(y_test, y_test_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_test_pred, average='weighted'):.4f}")

# 9.2 Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title(f"Confusion Matrix ({best_model_name})")
plt.ylabel('Gerçek Sınıf')
plt.xlabel('Tahmin Edilen Sınıf')
plt.show()

# 9.3 ROC Eğrisi (Multiclass)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green']
for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_test_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'Sınıf {i} ROC (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Yorumlanabilirlik için Ham Veri üzerinde eğitilmiş Random Forest modelini kullanıyoruz
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# SHAP Explainer Kurulumu
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

print("\n--- SHAP Summary Plot (Dot) ---")
# Class 1 (veya 0, 2) için özellikleri gösterir. Genelde summary_plot tüm sınıfları gösterir.
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

print("\n--- SHAP Bar Plot (Özellik Önemi) ---")
shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar")

# --- 10.2 ---
print("PCA ve LDA için SHAP Analizi Başlatılıyor...\n")
import shap
import matplotlib.pyplot as plt


def safe_shap_plot(model, X_train, X_test, feature_names, title):
    try:
        print(f"--- {title} Hesaplanıyor ---")
        # Explainer Kurulumu
        masker = shap.maskers.Independent(data=X_train)
        explainer = shap.LinearExplainer(model, masker)
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list):
            # Liste ise ilk sınıfı al (Class 0)
            vals_to_plot = shap_values[0]
        elif len(shap_values.shape) == 3:
            # 3 Boyutlu array ise (Örnek, Özellik, Sınıf) -> İlk sınıfı al
            vals_to_plot = shap_values[:, :, 0]
        else:
            # Zaten düzgünse
            vals_to_plot = shap_values

        # Çizim
        plt.figure()
        shap.summary_plot(vals_to_plot, X_test, feature_names=feature_names, show=False)
        plt.title(f"SHAP: {title} (Sınıf 0 Etkisi)")
        plt.show()

    except Exception as e:
        print(f"⚠️ {title} {e}")
        print("")

# 1. PCA İÇİN
model_pca = LogisticRegression().fit(X_train_pca, y_train)
safe_shap_plot(model_pca, X_train_pca, X_test_pca, ['PC1', 'PC2'], "PCA Temsili")

# 2. LDA İÇİN
model_lda = LogisticRegression().fit(X_train_lda, y_train)
safe_shap_plot(model_lda, X_train_lda, X_test_lda, ['LD1', 'LD2'], "LDA Temsili")