import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import warnings
warnings.filterwarnings("ignore")

start = time.time()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "UrbanSound8K.csv")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError("UrbanSound8K.csv not found in project folder.")

OUTPUT_DIR = os.path.join(BASE_DIR, "Urban_Audio_Intelligence_Output")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

plt.style.use("dark_background")

df = pd.read_csv(INPUT_FILE)
df.columns = [c.lower().replace(" ", "_") for c in df.columns]

if "class" not in df.columns:
    raise ValueError("Expected 'class' column not found.")

target_col = "class"

le = LabelEncoder()
df["class_encoded"] = le.fit_transform(df[target_col])

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "class_encoded"]
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# ---------------- Class Distribution ----------------
fig, ax = plt.subplots(figsize=(14,8))
sns.countplot(data=df, x=target_col, order=df[target_col].value_counts().index, ax=ax)
ax.set_title("Class Distribution of Environmental Sounds")
ax.set_xlabel("Sound Category")
ax.set_ylabel("Number of Samples")
plt.xticks(rotation=45)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "class_distribution.png"), dpi=300)
plt.close()

# ---------------- Scaling ----------------
scaler = StandardScaler()
scaled = scaler.fit_transform(df[numeric_cols])
pickle.dump(scaler, open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb"))

# ---------------- PCA ----------------
pca = PCA(n_components=min(5, len(numeric_cols)))
pca_data = pca.fit_transform(scaled)
pickle.dump(pca, open(os.path.join(MODEL_DIR, "pca.pkl"), "wb"))

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
ax.set_title("Cumulative Explained Variance (PCA)")
ax.set_xlabel("Number of Principal Components")
ax.set_ylabel("Cumulative Explained Variance Ratio")
ax.set_ylim(0,1.05)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "pca_variance.png"), dpi=300)
plt.close()

if pca_data.shape[1] >= 2:
    fig, ax = plt.subplots(figsize=(10,8))
    scatter = ax.scatter(
        pca_data[:,0],
        pca_data[:,1],
        c=df["class_encoded"],
        cmap="tab10",
        alpha=0.7
    )
    ax.set_title("PCA Projection (PC1 vs PC2)")
    ax.set_xlabel(f"PC1 ({round(pca.explained_variance_ratio_[0]*100,2)}% variance)")
    ax.set_ylabel(f"PC2 ({round(pca.explained_variance_ratio_[1]*100,2)}% variance)")
    ax.grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=ax, label="Encoded Class")
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "pca_projection.png"), dpi=300)
    plt.close()

# ---------------- Clustering ----------------
kmeans = KMeans(n_clusters=df[target_col].nunique(), random_state=42, n_init=20)
clusters = kmeans.fit_predict(scaled)
sil_score = silhouette_score(scaled, clusters)

# ---------------- Feature Importance ----------------
mi = mutual_info_classif(df[numeric_cols], df["class_encoded"])
importance = pd.Series(mi, index=numeric_cols).sort_values()

fig, ax = plt.subplots(figsize=(10,8))
importance.plot(kind="barh", ax=ax)
ax.set_title("Feature Relevance (Mutual Information)")
ax.set_xlabel("Mutual Information Score")
ax.set_ylabel("Feature")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "feature_relevance.png"), dpi=300)
plt.close()

# ---------------- Train Model ----------------
X_train, X_test, y_train, y_test = train_test_split(
    scaled,
    df["class_encoded"],
    test_size=0.2,
    random_state=42,
    stratify=df["class_encoded"]
)

model = RandomForestClassifier(n_estimators=400, random_state=42)
model.fit(X_train, y_train)
pickle.dump(model, open(os.path.join(MODEL_DIR, "urban_audio_classifier.pkl"), "wb"))

y_pred = model.predict(X_test)

labels_present = np.unique(np.concatenate([y_test, y_pred]))
class_names = [str(le.inverse_transform([l])[0]) for l in labels_present]

report = classification_report(
    y_test,
    y_pred,
    labels=labels_present,
    target_names=class_names,
    zero_division=0
)

formatted_report = report.replace("\n", "<br/>")

cm = confusion_matrix(y_test, y_pred, labels=labels_present)

fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="viridis",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=ax
)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
fig.tight_layout()
fig.savefig(os.path.join(CHART_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

execution_time = round(time.time() - start, 2)

# ---------------- PDF REPORT ----------------
styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    name="title",
    fontSize=26,
    leading=32,
    alignment=1,
    textColor=HexColor("#22d3ee"),
    spaceAfter=30
)

doc = SimpleDocTemplate(os.path.join(OUTPUT_DIR, "Urban_Audio_Intelligence_Report.pdf"))
elements = []

elements.append(Paragraph("Urban Environmental Sound Classification Intelligence Report", title_style))

summary = f"""
Dataset File: UrbanSound8K.csv<br/>
Total Audio Samples: {len(df)}<br/>
Number of Sound Classes: {df[target_col].nunique()}<br/>
Clustering Silhouette Score: {round(sil_score,3)}<br/>
Execution Time: {execution_time} seconds<br/><br/>
Model Performance Summary:<br/><br/>
{formatted_report}
"""

elements.append(Paragraph(summary, styles["Normal"]))
elements.append(PageBreak())

charts = sorted(os.listdir(CHART_DIR))
for chart in charts:
    elements.append(Paragraph(chart.replace("_"," ").replace(".png","").title(), styles["Heading2"]))
    elements.append(Image(os.path.join(CHART_DIR, chart), width=6*inch, height=4*inch))
    elements.append(PageBreak())

doc.build(elements)

print("Urban Audio Intelligence Pipeline Complete")
print("Execution Time:", execution_time)