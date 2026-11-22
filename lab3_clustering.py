# -*- coding: utf-8 -*-
"""Лабораторная работа 3. Полный листинг.

Это тот же код, что в ячейках lab3_clustering.ipynb, собранный в один файл,
чтобы его можно было открыть и рассказать по шагам.
Рядом должен лежать data.csv. Запуск из этой папки:

    python lab3_clustering.py

Графики откроются окнами. Готовый отчёт с уже посчитанными рисунками — в ноутбуке.
"""

# ========================================================================
# Лабораторная работа 3. Кластеризация аудиопризнаков треков Spotify
# ========================================================================


# ========================================================================
# 2. Описание датасета
# ========================================================================


import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.titlesize": 12,
    "font.family": "DejaVu Sans",
    "savefig.facecolor": "white",
})

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()
FIGURES = HERE / "figures"
RESULTS = HERE / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def save_show(name):
    plt.gcf().savefig(FIGURES / name, bbox_inches="tight")
    print("Сохранён график:", name)
    plt.show()


def save_table(frame, name):
    frame.to_csv(RESULTS / name, encoding="utf-8-sig")
    print("Сохранена таблица:", name)


RANDOM_STATE = 42
SAMPLE_SIZE = 6000
K_RANGE = range(2, 11)
AUDIO_FEATURES = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
    "duration_ms",
]
CLUSTER_ORDER = ["Акустические", "Энергичные", "Речевые"]
PALETTE = {
    "Акустические": "#4C78A8",
    "Энергичные": "#F58518",
    "Речевые": "#54A24B",
}

tracks = pd.read_csv("data.csv")
print(f"Размерность: {tracks.shape[0]} строк, {tracks.shape[1]} столбцов")
print(f"Дубликаты id: {int(tracks['id'].duplicated().sum())}")
print(f"Пропуски: {int(tracks.isna().sum().sum())}")
display(tracks.dtypes.rename("тип").to_frame())

summary = tracks[AUDIO_FEATURES].describe().T
summary["skew"] = tracks[AUDIO_FEATURES].skew()
summary["excess_kurtosis"] = tracks[AUDIO_FEATURES].kurtosis()
summary = summary.rename(columns={
    "count": "n",
    "mean": "среднее",
    "std": "СКО",
    "min": "мин",
    "25%": "Q1",
    "50%": "медиана",
    "75%": "Q3",
    "max": "макс",
    "skew": "асимметрия",
    "excess_kurtosis": "эксцесс",
})
display(summary.round(3))
save_table(summary.round(3), "описательные_статистики.csv")

normal_sample = tracks[AUDIO_FEATURES].sample(5000, random_state=RANDOM_STATE)
normality_rows = []
for feature in AUDIO_FEATURES:
    statistic, p_value = stats.shapiro(normal_sample[feature])
    normality_rows.append({
        "признак": feature,
        "асимметрия, n=170653": tracks[feature].skew(),
        "эксцесс, n=170653": tracks[feature].kurtosis(),
        "Шапиро–Уилк W, n=5000": statistic,
        "p-value, n=5000": p_value,
    })
normality = pd.DataFrame(normality_rows)
save_table(normality, "нормальность.csv")
display(normality.style.format({
    "асимметрия, n=170653": "{:.3f}",
    "эксцесс, n=170653": "{:.3f}",
    "Шапиро–Уилк W, n=5000": "{:.3f}",
    "p-value, n=5000": "{:.2e}",
}).hide(axis="index"))

fig, axes = plt.subplots(2, 5, figsize=(14, 5.8))
for axis, feature in zip(axes.ravel(), AUDIO_FEATURES):
    values = tracks[feature].to_numpy()
    axis.hist(values, bins=40, density=True, color="#4C78A8", alpha=0.85, edgecolor="white")
    grid = np.linspace(values.min(), values.max(), 300)
    axis.plot(grid, stats.norm.pdf(grid, values.mean(), values.std(ddof=0)), color="#E45756", lw=1.4)
    axis.set_title(feature)
fig.suptitle("Распределения аудиопризнаков и кривая нормального закона с теми же средним и СКО", y=1.02)
fig.tight_layout()
save_show("01_распределения.png")

# Не информационная значимость: дисперсия duration_ms считается в мс²,
# поэтому доля сырой дисперсии — артефакт единиц измерения.
raw_variance_share = tracks[AUDIO_FEATURES].var()
raw_variance_share = (raw_variance_share / raw_variance_share.sum() * 100).rename(
    "доля сырой дисперсии, %, артефакт единиц"
)
display(raw_variance_share.to_frame().style.format("{:.6f}"))

bounded = ["acousticness", "danceability", "energy", "instrumentalness", "liveness", "speechiness", "valence"]
spread = tracks[bounded].agg(["mean", "std"]).T
spread["СКО / среднее"] = spread["std"] / spread["mean"]
spread = spread.rename(columns={"mean": "среднее", "std": "СКО"})
display(spread.round(3))

correlation = tracks[AUDIO_FEATURES].corr()
fig, axis = plt.subplots(figsize=(8.2, 6.4))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="vlag", center=0, ax=axis, annot_kws={"size": 8})
axis.set_title("Корреляция аудиопризнаков")
fig.tight_layout()
save_show("02_корреляция.png")

def iqr_fence_share(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    outside = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
    return pd.Series({
        "Q1": q1,
        "Q3": q3,
        "нижняя граница": q1 - 1.5 * iqr,
        "верхняя граница": q3 + 1.5 * iqr,
        "за границей, шт.": int(outside.sum()),
        "за границей, доля": outside.mean(),
    })

iqr_table = tracks[AUDIO_FEATURES].apply(iqr_fence_share).T
save_table(iqr_table, "выбросы_iqr.csv")
display(iqr_table.style.format({
    "Q1": "{:.3f}",
    "Q3": "{:.3f}",
    "нижняя граница": "{:.3f}",
    "верхняя граница": "{:.3f}",
    "за границей, шт.": "{:.0f}",
    "за границей, доля": "{:.1%}",
}))

fig, axes = plt.subplots(2, 5, figsize=(14, 5.8))
for axis, feature in zip(axes.ravel(), AUDIO_FEATURES):
    axis.boxplot(tracks[feature], showfliers=False, widths=0.6)
    axis.set_title(feature)
    axis.set_xticks([])
fig.suptitle("Ящики с усами без отдельных точек: форма распределения, а не шкала хвоста", y=1.02)
fig.tight_layout()
save_show("03_ящики_с_усами.png")

print(
    "duration_ms: медиана "
    f"{tracks['duration_ms'].median():.0f} мс, максимум {tracks['duration_ms'].max():.0f} мс "
    "(около 90 минут — аудиокнига, не трек); "
    f"tempo = 0 у {(tracks['tempo'] <= 0).sum()} строк, "
    f"danceability = 0 у {(tracks['danceability'] <= 0).sum()} строк "
    "(дефект Spotify API, не пустые треки)."
)


# ========================================================================
# 3. Предобработка данных
# ========================================================================

invalid_tempo = tracks["tempo"] <= 0
invalid_duration = ~tracks["duration_ms"].between(30_000, 600_000)
print(f"tempo <= 0: {int(invalid_tempo.sum())}")
print(f"длительность вне [30 с; 10 мин]: {int(invalid_duration.sum())}")
print(f"пересечение этих условий: {int((invalid_tempo & invalid_duration).sum())}")

clean = tracks.loc[~(invalid_tempo | invalid_duration)].copy()
print(f"удалено {len(tracks) - len(clean)} строк ({(len(tracks) - len(clean)) / len(tracks):.1%})")
print(f"осталось {len(clean)} строк")

sample = clean.sample(SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
comparison = pd.DataFrame({
    "вся очищенная таблица": clean[AUDIO_FEATURES].mean(),
    "подвыборка 6000": sample[AUDIO_FEATURES].mean(),
})
comparison["относительное отклонение"] = (
    comparison["подвыборка 6000"] - comparison["вся очищенная таблица"]
) / comparison["вся очищенная таблица"]
display(comparison.round(4))
save_table(comparison.round(4), "подвыборка_средние.csv")

scaler = StandardScaler()
X = scaler.fit_transform(sample[AUDIO_FEATURES])
scaled = pd.DataFrame(X, columns=AUDIO_FEATURES)
print("Средние после StandardScaler:", np.round(scaled.mean().to_numpy(), 8))
print("СКО после StandardScaler:", np.round(scaled.std(ddof=0).to_numpy(), 8))

pair_features = ["acousticness", "energy", "loudness", "danceability", "valence", "speechiness"]
pair_view = scaled[pair_features].sample(700, random_state=RANDOM_STATE)
graph = sns.pairplot(pair_view, corner=True, diag_kind="hist", plot_kws={"s": 8, "alpha": 0.35, "color": "#4C78A8"}, height=1.45)
graph.fig.suptitle(
    "Матрица рассеяния: 6 признаков из 10, стандартизованные координаты",
    y=1.02,
)
graph.fig.savefig(FIGURES / "04_матрица_рассеяния.png", bbox_inches="tight")
print("Сохранён график:", "04_матрица_рассеяния.png")
plt.show()


# ========================================================================
# 4. Ход работы
# ========================================================================

kmeans_rows = []
for k in K_RANGE:
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = model.fit_predict(X)
    kmeans_rows.append({
        "k": k,
        "инерция": model.inertia_,
        "силуэт": silhouette_score(X, labels),
        "Калинский–Харабаш": calinski_harabasz_score(X, labels),
        "Дэвис–Болдин": davies_bouldin_score(X, labels),
    })
kmeans_k = pd.DataFrame(kmeans_rows)

print("Матрица связей Уорда...")
linkage_matrix = linkage(X, method="ward")
hier_rows = []
for k in K_RANGE:
    labels = fcluster(linkage_matrix, t=k, criterion="maxclust")
    hier_rows.append({
        "k": k,
        "силуэт": silhouette_score(X, labels),
        "Калинский–Харабаш": calinski_harabasz_score(X, labels),
        "Дэвис–Болдин": davies_bouldin_score(X, labels),
    })
hier_k = pd.DataFrame(hier_rows)

save_table(kmeans_k, "подбор_k_kmeans.csv")
save_table(hier_k, "подбор_k_уорд.csv")
display(kmeans_k.style.format({
    "инерция": "{:.1f}",
    "силуэт": "{:.3f}",
    "Калинский–Харабаш": "{:.1f}",
    "Дэвис–Болдин": "{:.3f}",
}).hide(axis="index").set_caption("K-means"))
display(hier_k.style.format({
    "силуэт": "{:.3f}",
    "Калинский–Харабаш": "{:.1f}",
    "Дэвис–Болдин": "{:.3f}",
}).hide(axis="index").set_caption("Иерархическая кластеризация, связь Уорда"))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(kmeans_k["k"], kmeans_k["инерция"], marker="o", color="#4C78A8")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Инерция")
axes[0].set_title("Метод локтя, K-means")
axes[0].set_xticks(list(K_RANGE))

axes[1].plot(kmeans_k["k"], kmeans_k["силуэт"], marker="o", label="K-means")
axes[1].plot(hier_k["k"], hier_k["силуэт"], marker="s", linestyle="--", label="Уорд")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Коэффициент силуэта")
axes[1].set_title("Силуэтный анализ")
axes[1].set_xticks(list(K_RANGE))
axes[1].legend()
fig.tight_layout()
save_show("05_локоть_и_силуэт.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(kmeans_k["k"], kmeans_k["Калинский–Харабаш"], marker="o", label="K-means")
axes[0].plot(hier_k["k"], hier_k["Калинский–Харабаш"], marker="s", linestyle="--", label="Уорд")
axes[0].set_title("Индекс Калинского–Харабаша")
axes[1].plot(kmeans_k["k"], kmeans_k["Дэвис–Болдин"], marker="o", label="K-means")
axes[1].plot(hier_k["k"], hier_k["Дэвис–Болдин"], marker="s", linestyle="--", label="Уорд")
axes[1].set_title("Индекс Дэвиса–Болдина")
for axis in axes:
    axis.set_xlabel("k")
    axis.set_xticks(list(K_RANGE))
    axis.legend()
fig.tight_layout()
save_show("06_индексы_качества.png")


# ========================================================================
# 4.2. Итоговые модели и метрики качества
# ========================================================================

def cluster_names(labels, frame):
    scores = frame.assign(cluster=labels).groupby("cluster")[["speechiness", "energy"]].mean()
    speech = scores["speechiness"].idxmax()
    rest = scores.drop(index=speech)
    energetic = rest["energy"].idxmax()
    acoustic = rest["energy"].idxmin()
    return {speech: "Речевые", energetic: "Энергичные", acoustic: "Акустические"}


def geometry(features, labels):
    rows = []
    centers = []
    names = []
    sil_values = silhouette_samples(features, labels)
    for name in CLUSTER_ORDER:
        mask = labels == name
        points = features[mask]
        center = points.mean(axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        rows.append({
            "кластер": name,
            "размер": int(mask.sum()),
            "доля": mask.mean(),
            "средняя дистанция до центра": distances.mean(),
            "медианная дистанция до центра": np.median(distances),
            "максимальная дистанция до центра": distances.max(),
            "сумма квадратов дистанций": np.sum(distances ** 2),
            "средний силуэт": sil_values[mask].mean(),
        })
        centers.append(center)
        names.append(name)
    centers = np.vstack(centers)
    distance_matrix = pd.DataFrame(
        np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2),
        index=names,
        columns=names,
    )
    return pd.DataFrame(rows), distance_matrix


kmeans_model = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE)
kmeans_raw = kmeans_model.fit_predict(X)
kmeans_map = cluster_names(kmeans_raw, sample)
kmeans_labels = pd.Series(kmeans_raw).map(kmeans_map).to_numpy()

hier_raw = fcluster(linkage_matrix, t=3, criterion="maxclust")
hier_map = cluster_names(hier_raw, sample)
hier_labels = pd.Series(hier_raw).map(hier_map).to_numpy()

kmeans_within, kmeans_between = geometry(X, kmeans_labels)
hier_within, hier_between = geometry(X, hier_labels)

metrics = pd.DataFrame({
    "K-means": [
        silhouette_score(X, kmeans_labels),
        calinski_harabasz_score(X, kmeans_labels),
        davies_bouldin_score(X, kmeans_labels),
        kmeans_model.inertia_,
    ],
    "Уорд": [
        silhouette_score(X, hier_labels),
        calinski_harabasz_score(X, hier_labels),
        davies_bouldin_score(X, hier_labels),
        np.nan,
    ],
}, index=["Силуэт", "Калинский–Харабаш", "Дэвис–Болдин", "Инерция"])
save_table(metrics, "сравнение_методов.csv")
display(metrics.style.format("{:.3f}", na_rep="—"))
agreement_metrics = pd.DataFrame({
    "значение": [
        adjusted_rand_score(kmeans_labels, hier_labels),
        normalized_mutual_info_score(kmeans_labels, hier_labels),
        (kmeans_labels == hier_labels).mean(),
    ]
}, index=["ARI двух разбиений", "NMI двух разбиений", "доля совпавших имён"])
display(agreement_metrics.round(3))
save_table(agreement_metrics.round(3), "согласованность_методов.csv")

print("K-means: размер, компактность, силуэт по кластерам")
display(kmeans_within.style.format({
    "доля": "{:.1%}",
    "средняя дистанция до центра": "{:.3f}",
    "медианная дистанция до центра": "{:.3f}",
    "максимальная дистанция до центра": "{:.3f}",
    "сумма квадратов дистанций": "{:.1f}",
    "средний силуэт": "{:.3f}",
}).hide(axis="index"))
print("Расстояния между центрами K-means в стандартизованном пространстве")
save_table(kmeans_within, "геометрия_kmeans.csv")
display(kmeans_between.round(3))
save_table(kmeans_between.round(3), "расстояния_центров_kmeans.csv")

print("Уорд: размер, компактность, силуэт по кластерам")
display(hier_within.style.format({
    "доля": "{:.1%}",
    "средняя дистанция до центра": "{:.3f}",
    "медианная дистанция до центра": "{:.3f}",
    "максимальная дистанция до центра": "{:.3f}",
    "сумма квадратов дистанций": "{:.1f}",
    "средний силуэт": "{:.3f}",
}).hide(axis="index"))
print("Расстояния между центрами Уорда в стандартизованном пространстве")
save_table(hier_within, "геометрия_уорд.csv")
display(hier_between.round(3))
save_table(hier_between.round(3), "расстояния_центров_уорд.csv")

agreement = (kmeans_labels == hier_labels).mean()
print(f"Доля объектов с одинаковым именем кластера: {agreement:.1%}")
crosstab = pd.crosstab(
    pd.Series(kmeans_labels, name="K-means"),
    pd.Series(hier_labels, name="Уорд"),
).reindex(index=CLUSTER_ORDER, columns=CLUSTER_ORDER)
display(crosstab)
save_table(crosstab, "сопряжённость.csv")


# ========================================================================
# 4.3. Влияние параметра k и устойчивость
# ========================================================================

reference = kmeans_labels
stability = []
for seed in range(5):
    labels = KMeans(n_clusters=3, n_init=10, random_state=seed).fit_predict(X)
    named = pd.Series(labels).map(cluster_names(labels, sample)).to_numpy()
    stability.append({
        "random_state": seed,
        "ARI с опорным решением": adjusted_rand_score(reference, named),
    })
stability_table = pd.DataFrame(stability)
save_table(stability_table, "устойчивость_инициализации.csv")
display(stability_table.style.format({"ARI с опорным решением": "{:.3f}"}).hide(axis="index"))

full_scaler = StandardScaler()
X_full = full_scaler.fit_transform(clean[AUDIO_FEATURES])
full_raw = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE).fit_predict(X_full)
full_named = pd.Series(full_raw).map(cluster_names(full_raw, clean)).to_numpy()
full_profile = clean.assign(кластер=full_named).groupby("кластер")[AUDIO_FEATURES].mean().reindex(CLUSTER_ORDER)
full_sizes = pd.Series(full_named).value_counts().reindex(CLUSTER_ORDER)
full_profile.insert(0, "размер", full_sizes.to_numpy())
print("K-means по всей очищенной таблице")
display(full_profile.round(3))
save_table(full_profile.round(3), "центры_полная_выборка.csv")
print("Доли на полной таблице:")
display((pd.Series(full_named).value_counts(normalize=True).reindex(CLUSTER_ORDER) * 100).round(1).rename("доля, %").to_frame())

core_features = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "loudness", "speechiness", "valence",
]
X_core = StandardScaler().fit_transform(sample[core_features])
core_raw = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE).fit_predict(X_core)
core_named = pd.Series(core_raw).map(cluster_names(core_raw, sample)).to_numpy()
core_check = pd.DataFrame({
    "модель": ["все 10 признаков", "без liveness, tempo, duration_ms"],
    "силуэт": [
        silhouette_score(X, kmeans_labels),
        silhouette_score(X_core, core_named),
    ],
    "Калинский–Харабаш": [
        calinski_harabasz_score(X, kmeans_labels),
        calinski_harabasz_score(X_core, core_named),
    ],
    "Дэвис–Болдин": [
        davies_bouldin_score(X, kmeans_labels),
        davies_bouldin_score(X_core, core_named),
    ],
    "ARI с основной моделью": [
        1.0,
        adjusted_rand_score(kmeans_labels, core_named),
    ],
})
display(core_check.round(3))
save_table(core_check.round(3), "проверка_состава_признаков.csv")
display(pd.Series(core_named).value_counts().reindex(CLUSTER_ORDER).rename("размер без трёх признаков").to_frame())


# ========================================================================
# 5. Визуализация результатов
# ========================================================================

fig, axis = plt.subplots(figsize=(11, 4.8))
cut_height = (linkage_matrix[-2, 2] + linkage_matrix[-3, 2]) / 2
dendrogram(
    linkage_matrix,
    truncate_mode="lastp",
    p=20,
    leaf_rotation=90,
    color_threshold=cut_height,
    ax=axis,
)
axis.axhline(cut_height, color="#333333", linestyle="--", linewidth=0.9)
axis.text(0.2, cut_height + 3, "разрез на 3 кластера", fontsize=9, color="#333333")
axis.set_title("Дендрограмма Уорда, последние 20 слияний")
axis.set_xlabel("Число объектов в схлопнутой ветви")
axis.set_ylabel("Расстояние слияния")
fig.tight_layout()
save_show("07_дендрограмма.png")

pca = PCA(n_components=2, random_state=RANDOM_STATE)
xy = pca.fit_transform(X)
loadings = pd.DataFrame(
    pca.components_.T,
    index=AUDIO_FEATURES,
    columns=["PC1", "PC2"],
)
explained = pca.explained_variance_ratio_
print(f"Доля дисперсии: PC1 {explained[0]:.1%}, PC2 {explained[1]:.1%}, вместе {explained.sum():.1%}")
display(loadings.round(3))
save_table(loadings.round(3), "нагрузки_pca.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for axis, component in zip(axes, ["PC1", "PC2"]):
    ordered = loadings[component].sort_values()
    colors = ["#E45756" if value < 0 else "#4C78A8" for value in ordered]
    axis.barh(ordered.index, ordered.values, color=colors)
    axis.axvline(0, color="black", lw=0.6)
    axis.set_title(component)
fig.suptitle("Нагрузки признаков на первые две главные компоненты", y=1.02)
fig.tight_layout()
save_show("08_нагрузки_pca.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
for axis, labels, title in (
    (axes[0], kmeans_labels, "K-means"),
    (axes[1], hier_labels, "Уорд"),
):
    for name in ["Акустические", "Энергичные", "Речевые"]:
        mask = labels == name
        axis.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=9,
            alpha=0.65,
            c=PALETTE[name],
            label=name,
            linewidths=0,
        )
    axis.set_title(title)
    axis.set_xlabel(f"PC1 ({explained[0]:.1%} дисперсии)")
    axis.set_ylabel(f"PC2 ({explained[1]:.1%} дисперсии)")
    axis.legend(markerscale=2, frameon=True)
fig.suptitle("Кластеры в проекции на две главные компоненты", y=1.02)
fig.tight_layout()
save_show("09_кластеры_pca.png")

profile_columns = AUDIO_FEATURES + ["year", "popularity", "explicit", "mode"]


def original_profile(frame, labels):
    grouped = frame.assign(кластер=labels).groupby("кластер")[profile_columns].mean().reindex(CLUSTER_ORDER)
    sizes = pd.Series(labels).value_counts().reindex(CLUSTER_ORDER)
    grouped.insert(0, "размер", sizes.to_numpy())
    return grouped


print("Центры K-means в исходных единицах")
kmeans_centers = original_profile(sample, kmeans_labels).round(3)
display(kmeans_centers)
save_table(kmeans_centers, "центры_kmeans.csv")
print("Центры Уорда в исходных единицах")
hier_centers = original_profile(sample, hier_labels).round(3)
display(hier_centers)
save_table(hier_centers, "центры_уорд.csv")


def z_profile(frame, labels):
    means = frame.assign(кластер=labels).groupby("кластер")[AUDIO_FEATURES].mean().reindex(CLUSTER_ORDER)
    return (means - frame[AUDIO_FEATURES].mean()) / frame[AUDIO_FEATURES].std(ddof=0)


fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2))
for axis, labels, title in (
    (axes[0], kmeans_labels, "K-means"),
    (axes[1], hier_labels, "Уорд"),
):
    sns.heatmap(z_profile(sample, labels), cmap="vlag", center=0, annot=True, fmt=".2f", ax=axis)
    axis.set_title(f"{title}: центр кластера в долях СКО относительно подвыборки")
fig.tight_layout()
save_show("10_профили_кластеров.png")

overall = X.mean(axis=0)
total_ss = ((X - overall) ** 2).sum(axis=0)
between_ss = np.zeros(X.shape[1])
for name in CLUSTER_ORDER:
    mask = kmeans_labels == name
    between_ss += mask.sum() * (X[mask].mean(axis=0) - overall) ** 2
eta = pd.Series(between_ss / total_ss, index=AUDIO_FEATURES, name="доля межкластерной дисперсии").sort_values(ascending=False)
display(eta.round(3).to_frame())
save_table(eta.round(3).to_frame(), "значимость_признаков.csv")

labeled = sample.assign(кластер=kmeans_labels)
columns = ["кластер", "year", "name", "artists", "speechiness", "energy", "acousticness", "popularity"]
examples = []
for name in CLUSTER_ORDER:
    mask = labeled["кластер"].to_numpy() == name
    center = X[mask].mean(axis=0)
    distance = np.linalg.norm(X[mask] - center, axis=1)
    nearest = labeled.index.to_numpy()[mask][np.argsort(distance)[:5]]
    examples.append(labeled.loc[nearest, columns])
display(pd.concat(examples, ignore_index=True))

speech_year = labeled.loc[labeled["кластер"] == "Речевые", "year"]
print(speech_year.describe().round(1).to_string())
print(f"доля речевых записей с year < 1950: {(speech_year < 1950).mean():.1%}")


# ========================================================================
# 6. Заключение
# ========================================================================


# ========================================================================
# 7. Список источников
# ========================================================================


# ========================================================================
# 8. Приложение
# ========================================================================

import matplotlib
import scipy
import sklearn

versions = pd.Series({
    "python": __import__("sys").version.split()[0],
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "scikit-learn": sklearn.__version__,
    "scipy": scipy.__version__,
    "matplotlib": matplotlib.__version__,
    "seaborn": sns.__version__,
})
display(versions.rename("версия").to_frame())
