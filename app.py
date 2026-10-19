"""Iris Species Classifier — Streamlit application.

Run with:  streamlit run app.py

The page is split into four tabs. Each tab is drawn by one render_* function
so the layout is easy to follow; all ML logic lives in src/.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.config import FEATURE_LABELS, FEATURES, SPECIES, SPECIES_COLORS
from src.data import load_iris_frame
from src.predict import (
    decision_grid,
    ensure_trained,
    load_metrics,
    load_model,
    out_of_range_warnings,
    predict_species,
)

PRESETS = {  # real rows from the dataset (0-based index)
    "Typical setosa": 0,
    "Typical versicolor": 50,
    "Typical virginica": 100,
    "Borderline case": 83,
}
PLOT_MARGIN = dict(l=10, r=10, t=10, b=10)
ASSETS = Path(__file__).resolve().parent / "assets"
SPECIES_PHOTOS = {  # illustrative photos, licences in assets/ATTRIBUTIONS.md
    "setosa": ("iris_setosa.jpg", "Radomil, CC BY-SA 3.0"),
    "versicolor": ("iris_versicolor.jpg", "Danielle Langlois, CC BY-SA 3.0"),
    "virginica": ("iris_virginica.jpg", "Frank Mayfield, CC BY-SA 2.0"),
}


@st.cache_resource(show_spinner="Preparing the model…")
def get_model_and_metrics():
    ensure_trained()
    return load_model(), load_metrics()


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_iris_frame()


def apply_preset(df: pd.DataFrame, row: int) -> None:
    for f in FEATURES:
        st.session_state[f"in_{f}"] = float(df.loc[row, f])


# --------------------------------------------------------------------------- #
# Predict tab
# --------------------------------------------------------------------------- #
def render_inputs(df: pd.DataFrame) -> dict:
    st.subheader("Flower measurements")
    st.caption("Load a real example from the dataset, or move the sliders.")
    b1, b2 = st.columns(2)
    for i, (name, row) in enumerate(PRESETS.items()):
        (b1 if i % 2 == 0 else b2).button(
            name, on_click=apply_preset, args=(df, row), width="stretch", key=f"preset_{row}"
        )
    return {f: st.slider(FEATURE_LABELS[f], 0.1, 10.0, step=0.1, key=f"in_{f}") for f in FEATURES}


def render_result(result: dict, values: dict, metrics: dict) -> None:
    species, conf = result["species"], result["confidence"]
    ranked = sorted(result["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
    if conf >= 0.85:
        note = "The model is confident in this prediction."
    elif conf >= 0.6:
        note = f"Fairly confident, but {ranked[1][0]} is also plausible."
    else:
        note = f"Uncertain: this flower sits between {ranked[0][0]} and {ranked[1][0]}."
    ui.result_card("Predicted species", f"Iris {species}", f"{conf:.1%} probability · {note}",
                   SPECIES_COLORS[species])

    fig = go.Figure(go.Bar(
        x=[p for _, p in ranked], y=[s for s, _ in ranked], orientation="h",
        marker_color=[SPECIES_COLORS[s] for s, _ in ranked],
        text=[f"{p:.1%}" for _, p in ranked], textposition="auto", insidetextfont=dict(color="white"),
    ))
    fig.update_layout(height=190, margin=PLOT_MARGIN, plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(range=[0, 1], tickformat=".0%", title="Model probability"),
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    st.caption("The three probabilities always add up to 100%. They come from the softmax of the model's "
               "three species scores and show how sure the model is, not a guarantee.")

    photo, credit = SPECIES_PHOTOS[species]
    p1, p2 = st.columns([1, 1.6], vertical_alignment="center")
    p1.image(str(ASSETS / photo), width="stretch")
    p2.markdown(f"**What *Iris {species}* looks like**")
    p2.caption(f"Illustrative photo only: the model never sees images, just your four measurements. "
               f"Photo: {credit} (Wikimedia Commons, see assets/ATTRIBUTIONS.md).")

    stats = metrics["train_stats"]
    warns = out_of_range_warnings(values, stats["min"], stats["max"])
    if warns:
        st.warning("**Outside the training data.** The model is extrapolating, so treat this "
                   "prediction with caution.\n\n" + "\n".join(f"- {w}" for w in warns))


def render_explanations(model, df: pd.DataFrame, values: dict, metrics: dict) -> None:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Where your flower falls on the model's decision map**")
        xs, ys, zz = decision_grid(model, values["sepal_length"], values["sepal_width"])
        c = [SPECIES_COLORS[s] for s in SPECIES]
        fig = go.Figure(go.Heatmap(
            x=xs, y=ys, z=zz, zmin=0, zmax=2, showscale=False, opacity=0.25, hoverinfo="skip",
            colorscale=[[0, c[0]], [0.333, c[0]], [0.333, c[1]], [0.666, c[1]], [0.666, c[2]], [1, c[2]]],
        ))
        for sp in SPECIES:
            sub = df[df["species"] == sp]
            fig.add_trace(go.Scatter(
                x=sub["petal_length"], y=sub["petal_width"], mode="markers", name=sp,
                marker=dict(color=SPECIES_COLORS[sp], size=7, line=dict(width=0.5, color="white")),
            ))
        fig.add_trace(go.Scatter(
            x=[values["petal_length"]], y=[values["petal_width"]], mode="markers", name="Your flower",
            marker=dict(symbol="star", size=20, color="#DB2777", line=dict(width=1.5, color="white")),
        ))
        fig.update_layout(height=420, margin=PLOT_MARGIN, plot_bgcolor="rgba(0,0,0,0)",
                          xaxis_title="Petal length (cm)", yaxis_title="Petal width (cm)",
                          legend=dict(orientation="h", y=-0.18))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Shaded regions show the species the model would predict for each petal size, with the "
            f"sepal measurements held at your values ({values['sepal_length']:g} cm × "
            f"{values['sepal_width']:g} cm). Dots are the 150 real flowers."
        )
    with c2:
        st.markdown("**Your flower compared with the average of each species**")
        means = metrics["train_stats"]["species_means"]
        labels = [FEATURE_LABELS[f].replace(" (cm)", "") for f in FEATURES]
        fig = go.Figure()
        for sp in SPECIES:
            fig.add_trace(go.Bar(name=f"{sp} (avg)", x=labels, y=[means[sp][f] for f in FEATURES],
                                 marker_color=SPECIES_COLORS[sp], opacity=0.75))
        fig.add_trace(go.Bar(name="Your flower", x=labels, y=[values[f] for f in FEATURES],
                             marker_color="#DB2777"))
        fig.update_layout(barmode="group", height=420, margin=PLOT_MARGIN, plot_bgcolor="rgba(0,0,0,0)",
                          yaxis_title="cm", legend=dict(orientation="h", y=-0.18))
        st.plotly_chart(fig, width="stretch")
        st.caption("Species averages come from the training split. Petal size separates the "
                   "species far more clearly than sepal size.")


# --------------------------------------------------------------------------- #
# Model card tab
# --------------------------------------------------------------------------- #
def render_model_card(metrics: dict) -> None:
    test, ds = metrics["test"], metrics["dataset"]
    st.subheader("How well does the model perform?")
    st.caption(
        "All numbers below are produced by `src/train.py` and read from `models/metrics.json`. "
        f"Test set: {ds['n_test']} flowers held out from training (stratified "
        f"{1 - ds['test_size']:.0%}/{ds['test_size']:.0%} split, random_state={ds['random_state']}). "
        f"Trained {metrics['trained_at_utc']} with scikit-learn {metrics['sklearn_version']}."
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{test['accuracy']:.1%}", border=True)
    m2.metric("Precision (macro)", f"{test['precision_macro']:.3f}", border=True)
    m3.metric("Recall (macro)", f"{test['recall_macro']:.3f}", border=True)
    m4.metric("F1-score (macro)", f"{test['f1_macro']:.3f}", border=True)
    lo, hi = test["accuracy_ci95_wilson"]
    st.info(
        f"With only {ds['n_test']} test flowers, each mistake changes accuracy by "
        f"{1 / ds['n_test']:.1%}. The model made {len(test['misclassified'])} mistake(s) on the test "
        f"set. A 95% confidence interval for the true accuracy is **{lo:.1%} – {hi:.1%}** (Wilson interval), "
        "so treat the single number as an estimate, and check the cross-validation spread below too."
    )

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown(f"**Model comparison ({metrics['cv_folds']}-fold cross-validation, training set)**")
        cv = metrics["cv_results"]
        names = list(cv)
        fig = go.Figure(go.Bar(
            # The mean is written into the row label: the x-axis starts at 0.8, so text placed
            # inside the bars would be clipped or hidden behind the error bars.
            x=[cv[n]["mean"] for n in names], y=[f"{n} · {cv[n]['mean']:.3f}" for n in names], orientation="h",
            error_x=dict(type="data", array=[cv[n]["std"] for n in names]),
            marker_color=["#7C3AED" if n == metrics["selected_model"] else "#C4B5FD" for n in names],
        ))
        fig.update_layout(height=300, margin=PLOT_MARGIN, plot_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(range=[0.8, 1.0], title="Mean accuracy ± std"),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Selected: **{metrics['selected_model']}**. {metrics['selection_rule']}")
    with b:
        st.markdown("**Confusion matrix (test set)**")
        fig = px.imshow(test["confusion_matrix"], x=SPECIES, y=SPECIES, text_auto=True,
                        color_continuous_scale="Purples",
                        labels=dict(x="Predicted", y="Actual", color="Count"))
        fig.update_layout(height=300, margin=PLOT_MARGIN, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    c, d = st.columns(2, gap="large")
    with c:
        st.markdown("**Which measurements matter? (permutation importance, test set)**")
        pi = metrics["permutation_importance"]
        order = sorted(FEATURES, key=lambda f: pi[f]["mean"])
        fig = go.Figure(go.Bar(
            x=[pi[f]["mean"] for f in order], y=[FEATURE_LABELS[f] for f in order], orientation="h",
            error_x=dict(type="data", array=[pi[f]["std"] for f in order]), marker_color="#7C3AED",
        ))
        fig.update_layout(height=260, margin=PLOT_MARGIN, plot_bgcolor="rgba(0,0,0,0)",
                          xaxis_title="Drop in test accuracy when the feature is shuffled")
        st.plotly_chart(fig, width="stretch")
    with d:
        st.markdown("**Per-species results (test set)**")
        rep = test["classification_report"]
        table = pd.DataFrame({sp: rep[sp] for sp in SPECIES}).T
        st.dataframe(table.style.format("{:.2f}", subset=["precision", "recall", "f1-score"])
                     .format("{:.0f}", subset=["support"]), width="stretch")
        if test["misclassified"]:
            st.markdown("**Test flowers the model got wrong**")
            st.dataframe(pd.DataFrame(test["misclassified"]), width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# How it works tab
# --------------------------------------------------------------------------- #
def render_how_it_works(metrics: dict) -> None:
    st.subheader("From measurements to a species")
    st.markdown(
        """
1. **Input**: four measurements in centimetres (sepal length and width, petal length and width).
2. **Validation**: each value must be a number in (0, 30] cm. Values outside the range seen in
   training are still predicted, but they trigger an extrapolation warning.
3. **Scaling** *(inside the model pipeline)*: each feature is standardised, `z = (x − mean) / std`,
   using the mean and standard deviation learned **from the training set only**.
4. **Classifier**: the selected model turns the four scaled numbers into a probability for each species.
5. **Output**: the species with the highest probability, the full probability distribution, and the
   visual explanations on the *Predict* tab.
        """
    )
    if metrics["selected_model"] == "Logistic Regression":
        st.markdown("**Multinomial logistic regression** computes one score per species and turns the "
                    "scores into probabilities with the softmax function:")
        st.latex(r"z_k = w_k^\top x + b_k \qquad P(y=k \mid x) = \frac{e^{z_k}}{\sum_{j} e^{z_j}}")
        st.markdown("The weights `w_k` are learned by minimising cross-entropy loss on the training "
                    "flowers. The decision boundaries are straight lines (hyperplanes) in the scaled "
                    "feature space, which makes the model simple, fast and easy to inspect.")
    st.markdown("#### What this model can and cannot tell you")
    st.markdown(
        """
- ✅ It can classify measurements that resemble the three species in Fisher's dataset
  (*setosa*, *versicolor*, *virginica*).
- ⚠️ It **always** answers with one of these three species, even for a flower of another species or
  a typo. It has no "none of the above" option; the extrapolation warning only partly addresses this.
- ⚠️ It learned from only 150 flowers, so it may not generalise to other populations or to
  measurements taken differently.
- ℹ️ The probabilities show how confident the model is. They do not guarantee the answer is correct.
        """
    )


# --------------------------------------------------------------------------- #
# Dataset tab
# --------------------------------------------------------------------------- #
def render_dataset(df: pd.DataFrame) -> None:
    st.subheader("Fisher's Iris dataset (1936)")
    st.markdown(
        "150 flowers, 50 of each species, with 4 measurements each and no missing values. Loaded "
        "from `sklearn.datasets.load_iris`, whose copy follows Fisher's original paper. Source: "
        "R. A. Fisher, *The use of multiple measurements in taxonomic problems*, Annals of Eugenics 7 "
        "(1936); UCI Machine Learning Repository (CC BY 4.0)."
    )
    st.markdown("**The three species** *(illustrative photos, not part of the data)*")
    cols = st.columns(3)
    for col, sp in zip(cols, SPECIES):
        photo, credit = SPECIES_PHOTOS[sp]
        col.image(str(ASSETS / photo), width="stretch")
        means = df[df["species"] == sp][FEATURES].mean()
        col.markdown(f"**Iris {sp}** · average petal {means['petal_length']:.1f} × {means['petal_width']:.1f} cm")
        col.caption(f"Photo: {credit}")
    st.caption("The **sepals** are the outer, leaf-like parts that protect the bud; the **petals** are the inner "
               "parts. In irises both are coloured, which is why both were measured. Photo sources and licences: "
               "`assets/ATTRIBUTIONS.md`.")
    fig = px.scatter_matrix(df, dimensions=FEATURES, color="species", color_discrete_map=SPECIES_COLORS,
                            labels={f: FEATURE_LABELS[f].replace(" (cm)", "") for f in FEATURES})
    fig.update_traces(diagonal_visible=False, marker=dict(size=4))
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Every pair of features. Setosa is separated from the others by petal size alone; "
               "versicolor and virginica overlap slightly, which is where the model's mistakes come from.")
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("**Summary statistics (cm)**")
        st.dataframe(df[FEATURES].describe().T.round(2), width="stretch")
    with right:
        st.markdown("**Raw data**")
        st.dataframe(df.drop(columns="target"), width="stretch", height=300)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Iris Species Classifier", page_icon="🌸", layout="wide")
    ui.inject_css()

    try:
        model, metrics = get_model_and_metrics()
        df = get_data()
    except Exception as exc:  # show a readable message instead of a stack trace
        st.error(f"The model could not be loaded or trained: {exc}")
        st.info("Run `python -m src.train` from the project folder, then reload this page.")
        st.stop()

    for f in FEATURES:  # default input: the first versicolor sample
        st.session_state.setdefault(f"in_{f}", float(df.loc[50, f]))

    ui.hero(
        "Iris Species Classifier",
        "Enter a flower's sepal and petal measurements. The model identifies the species and shows "
        "how it reached that decision.",
        [metrics["selected_model"], f"Test accuracy {metrics['test']['accuracy']:.1%}",
         "150 flowers · 3 species"],
    )
    tab_predict, tab_model, tab_how, tab_data = st.tabs(
        ["🔮 Predict", "📊 Model card", "🧠 How it works", "🗂️ Dataset"]
    )

    with tab_predict:
        col_in, col_out = st.columns([1, 1.35], gap="large")
        with col_in:
            values = render_inputs(df)
        with col_out:
            st.subheader("Prediction")
            try:
                result = predict_species(model, values)
            except ValueError as err:
                result = None
                st.error(f"Invalid input: {err}")
            if result is not None:
                render_result(result, values, metrics)
        if result is not None:
            st.divider()
            render_explanations(model, df, values, metrics)

    with tab_model:
        render_model_card(metrics)
    with tab_how:
        render_how_it_works(metrics)
    with tab_data:
        render_dataset(df)

    ui.footer("Iris Species Classifier · Arya Kumar · Educational machine-learning project")


main()
