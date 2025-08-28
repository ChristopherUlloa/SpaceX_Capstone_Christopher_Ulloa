import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import plotly.express as px
from collections import Counter
import numpy as np

def safe_write_image(fig, path_png):
    try:
        fig.write_image(path_png, scale=2)
        return path_png, None
    except Exception as e1:
        try:
            import kaleido
            kaleido.get_chrome_sync()
            fig.write_image(path_png, scale=2)
            return path_png, None
        except Exception as e2:
            alt = path_png.replace(".png", ".html")
            fig.write_html(alt, include_plotlyjs="cdn")
            return None, (alt, f"{e1}\n{e2}")


DATA = "data/launches.csv" if os.path.exists("data/launches.csv") else "data/sample_launches.csv"
df = pd.read_csv(DATA)
os.makedirs("imgs", exist_ok=True)


plt.figure(figsize=(8,4))
sns.countplot(x="LaunchSite", hue="Class", data=df)
plt.title("Success by Launch Site (Class=1 success)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("imgs/eda_success_by_site.png", dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7,4))
sns.scatterplot(x="FlightNumber", y="PayloadMass", hue="Class", data=df)
plt.title("Payload vs FlightNumber by Class")
plt.tight_layout()
plt.savefig("imgs/eda_payload_vs_flight.png", dpi=200, bbox_inches="tight")
plt.close()

fig1 = px.scatter(df, x="FlightNumber", y="PayloadMass", color="Class",
                  hover_data=["LaunchSite","Orbit","Serial"],
                  title="Plotly — FlightNumber vs Payload (by Class)")
png1, fb1 = safe_write_image(fig1, "imgs/plotly_scatter.png")

sr = df.groupby("LaunchSite")["Class"].mean().reset_index()
fig2 = px.bar(sr, x="LaunchSite", y="Class", title="Success Rate by Site")
fig2.update_yaxes(range=[0,1])
png2, fb2 = safe_write_image(fig2, "imgs/success_rate_by_site.png")

y = df["Class"].astype(int)
X = df.drop(columns=["Class","Date","Serial"], errors="ignore")

num = X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat = X.select_dtypes(include=["object"]).columns.tolist()

pre = ColumnTransformer([
    ("num", "passthrough", num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
])

def try_split_with_both_classes(X, y, test_size=0.2):
    """Intenta varios seeds sin estratificar hasta lograr train y test con ambas clases."""
    for seed in range(1000, 1030):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=None)
        if y_tr.nunique() == 2 and y_te.nunique() == 2:
            return X_tr, X_te, y_tr, y_te
    return None

split = try_split_with_both_classes(X, y, test_size=0.2)

if split is None:
    counts = y.value_counts()
    if len(counts) == 1:
        do_model = False
    else:
        do_model = True
        minority = counts.idxmin()
        need = max(0, 5 - counts[minority]) 
        df_bal = df.copy()
        if need > 0:
            df_min = df[df["Class"] == minority]
            df_bal = pd.concat([df_bal, df_min.sample(n=need, replace=True, random_state=42)], ignore_index=True)
        y_bal = df_bal["Class"].astype(int)
        X_bal = df_bal.drop(columns=["Class","Date","Serial"], errors="ignore")

        sp2 = try_split_with_both_classes(X_bal, y_bal, test_size=0.2)
        if sp2 is None:
            do_model = False
        else:
            X_train, X_test, y_train, y_test = sp2
else:
    do_model = True
    X_train, X_test, y_train, y_test = split

if do_model:
    pipe = Pipeline([("prep", pre), ("clf", DecisionTreeClassifier(max_depth=4, random_state=42))])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
else:

    cm = np.array([[0,0],[0,0]])

disp = ConfusionMatrixDisplay(cm, display_labels=[0,1])
disp.plot(values_format="d")
plt.title("Confusion Matrix — Decision Tree")
plt.savefig("imgs/confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close()

import folium
from folium.plugins import MarkerCluster
site_coords = {
    "CCAFS LC-40": (28.5618571, -80.577366),
    "CCAFS SLC-40": (28.5618571, -80.577366),
    "KSC LC-39A": (28.608389, -80.604333),
    "VAFB SLC-4E": (34.632093, -120.610829),
}
m = folium.Map(location=[28.56, -80.58], zoom_start=6)
cluster = MarkerCluster().add_to(m)
for _, r in df.iterrows():
    site = r["LaunchSite"]
    latlon = site_coords.get(site, (28.56,-80.58))
    txt = f"{site} | Orbit: {r['Orbit']} | Payload: {r['PayloadMass']} kg | Class: {r['Class']}"
    folium.Marker(location=latlon, popup=txt).add_to(cluster)
m.save("imgs/folium_map.html")

print("✅ Generado en /imgs:")
for f in [
    "eda_success_by_site.png",
    "eda_payload_vs_flight.png",
    "plotly_scatter.png" if png1 else "plotly_scatter.html",
    "success_rate_by_site.png" if png2 else "success_rate_by_site.html",
    "confusion_matrix.png",
    "folium_map.html",
]:
    print(" -", f)
