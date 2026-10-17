# =====================================================================
# evaluate.py - valutazione dei modelli: predizioni contro le tue etichette
#
# Parte da due tabelle:
#   - il dataset etichettato (preprocessing.attach_labels): comment_id, label_true,
#     text_clean, text_emoji, ...
#   - le predizioni (models.run_all): comment_id, model, version, label_pred
# e calcola, per ogni modello e per ogni versione del testo (text_clean = emoji
# grezze, text_emoji = emoji tradotte in parole):
#   1. le metriche (accuracy, precision, recall, F1) su tutti i commenti e su
#      sottoinsiemi (solo commenti con emoji, solo emoji, senza "neutro");
#   2. le matrici di confusione;
#   3. i test di McNemar: la traduzione delle emoji cambia davvero i risultati di
#      un modello? un modello e' davvero migliore di un altro?
#   4. esempi di commenti sbagliati, per l'analisi degli errori.
#
# Tutte le funzioni lavorano su DataFrame, quindi si possono usare anche dal
# notebook. Da terminale (dalla cartella principale del progetto, con il venv attivo):
#     python src/evaluate.py                  stampa tutte le tabelle
#     python src/evaluate.py --errors 5       aggiunge 5 commenti sbagliati per modello e versione
#     python src/evaluate.py --save           salva anche CSV e grafici in data/results/
# Tutte le opzioni: python src/evaluate.py --help
# =====================================================================

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


# ---------------------------------------------------------------------
# PERCORSI e COSTANTI
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = ROOT_DIR / "data" / "predictions" / "predizioni.csv"
RESULTS_DIR = ROOT_DIR / "data" / "results"       # cartella dentro data/, ignorata da git

# Le stesse etichette di models.py e preprocessing.py.
LABELS = ("positivo", "negativo", "neutro")

# Le due versioni del testo create da preprocessing.prepare_comments().
VERSION_CLEAN = "text_clean"
VERSION_EMOJI = "text_emoji"

# Sotto questa soglia di commenti le percentuali oscillano molto: la tabella lo segnala.
MIN_RELIABLE = 30


# Costruisce il dataset etichettato partendo dai file del progetto: legge i commenti di
# data/raw/, li prepara (prepare_comments) e ci unisce le tue etichette (attach_labels).
# `labels_csv` e' il file compilato a mano; se omesso usa data/labeled/etichettati_a_mano.csv.
# Serve a chi lancia lo script da terminale: dal notebook di solito il dataset c'e' gia'.
def load_labeled_dataset(labels_csv=None):
    try:
        import preprocessing as pp            # lanciato come: python src/evaluate.py
    except ImportError:
        from src import preprocessing as pp   # importato come modulo (es. dal notebook)

    dataset, _ = pp.prepare_comments(pp.load_raw_comments())
    return pp.attach_labels(dataset, labels_csv or pp.TO_LABEL_FILE)


# Legge il CSV delle predizioni creato da models.py (colonne comment_id, model, version,
# label_pred). Si ferma con un messaggio chiaro se il file manca o se le colonne non sono
# quelle attese, oppure se compare un'etichetta diversa da positivo/negativo/neutro.
def load_predictions(path=PREDICTIONS_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"Predizioni non trovate: {path}. "
                                f"Creale prima con: python src/models.py --dataset")
    predictions = pd.read_csv(path)

    missing = {"comment_id", "model", "version", "label_pred"} - set(predictions.columns)
    if missing:
        raise ValueError(f"Nel file delle predizioni mancano le colonne: {sorted(missing)}")
    unknown = set(predictions["label_pred"].dropna()) - set(LABELS)
    if unknown:
        raise ValueError(f"Etichette di predizione non valide: {sorted(unknown)}")
    return predictions


# Unisce le predizioni al dataset etichettato tramite comment_id e crea la tabella "lunga"
# su cui lavorano tutte le altre funzioni: una riga per (commento, modello, versione) con
#   label_true / label_pred : etichetta tua e del modello
#   correct                 : True se coincidono
#   has_emoji               : True se il commento contiene emoji (le due versioni del testo
#                             differiscono solo in quel caso)
# Le predizioni di commenti che non hai etichettato vengono scartate (con un avviso).
def merge_predictions(labeled, predictions):
    needed = ["comment_id", "label_true", "text_clean", "text_emoji"]
    missing = [c for c in needed if c not in labeled.columns]
    if missing:
        raise ValueError(f"Nel dataset etichettato mancano le colonne: {missing}")

    optional = [c for c in ("emoji_only", "user") if c in labeled.columns]
    merged = predictions.merge(labeled[needed + optional], on="comment_id", how="inner")

    dropped = len(predictions) - len(merged)
    if dropped:
        print(f"Attenzione: {dropped} predizioni scartate (commenti non presenti tra gli etichettati).")

    merged["correct"] = merged["label_pred"] == merged["label_true"]
    merged["has_emoji"] = merged["text_clean"] != merged["text_emoji"]
    return merged


# Divide i commenti in sottoinsiemi su cui calcolare le metriche, restituendo un dizionario
# nome -> tabella:
#   tutti         tutti i commenti etichettati
#   con_emoji     solo quelli con almeno un'emoji (gli unici dove le due versioni del testo
#                 differiscono: negli altri i risultati sono identici e "diluiscono" l'effetto)
#   solo_emoji    solo quelli fatti esclusivamente di emoji
#   senza_neutro  solo quelli che hai etichettato positivo o negativo: e' il confronto
#                 equo con feel_it, che e' un modello a 2 classi e non puo' mai dire "neutro"
# Un sottoinsieme viene incluso solo se non e' vuoto e se e' diverso da "tutti".
def make_subsets(merged):
    subsets = {"tutti": merged}

    def add(name, rows):
        if 0 < len(rows) < len(merged):
            subsets[name] = rows

    add("con_emoji", merged[merged["has_emoji"]])
    if "emoji_only" in merged.columns:
        add("solo_emoji", merged[merged["emoji_only"].astype(bool)])
    add("senza_neutro", merged[merged["label_true"] != "neutro"])
    return subsets


# Calcola le metriche per ogni coppia (modello, versione del testo). Colonne:
#   n                                numero di commenti valutati
#   accuracy                         quota di predizioni giuste
#   macro_precision / macro_recall / macro_f1
#                                    medie delle tre classi con lo STESSO peso ciascuna
#                                    (non risentono dello sbilanciamento tra classi)
#   f1_positivo / f1_negativo / f1_neutro
#                                    F1 di ogni classe (vuoto se in questi commenti non
#                                    ce n'e' nessuno)
# Le medie "macro" tengono conto solo delle classi che esistono davvero nelle tue
# etichette: se non hai commenti "neutro", la classe neutro non abbassa il risultato.
def compute_metrics(merged):
    rows = []
    for (model, version), g in merged.groupby(["model", "version"], sort=False):
        y_true, y_pred = g["label_true"], g["label_pred"]
        present = [label for label in LABELS if label in set(y_true)]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=present, average="macro", zero_division=0)
        _, _, f1_each, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(LABELS), average=None, zero_division=0)

        row = {
            "model": model,
            "version": version,
            "n": len(g),
            "accuracy": float(g["correct"].mean()),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
        }
        for label, value, n_true in zip(LABELS, f1_each, support):
            row[f"f1_{label}"] = float(value) if n_true > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# Calcola le metriche per tutti i sottoinsiemi di make_subsets() e le mette in un'unica
# tabella, con la colonna "subset" davanti.
def metrics_by_subset(merged):
    parts = []
    for name, rows in make_subsets(merged).items():
        part = compute_metrics(rows)
        part.insert(0, "subset", name)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


# Matrici di confusione, una per (modello, versione): dizionario (modello, versione) ->
# tabella 3x3 con le righe = etichetta vera (tua) e le colonne = predizione del modello.
# Sulla diagonale ci sono i commenti giusti; fuori diagonale si vede quali classi vengono
# scambiate tra loro.
def confusion_matrices(merged):
    matrices = {}
    for (model, version), g in merged.groupby(["model", "version"], sort=False):
        counts = confusion_matrix(g["label_true"], g["label_pred"], labels=list(LABELS))
        matrices[(model, version)] = pd.DataFrame(
            counts,
            index=[f"vero_{label}" for label in LABELS],
            columns=[f"pred_{label}" for label in LABELS],
        )
    return matrices


# Test di McNemar (versione esatta) per confrontare due serie di risposte sugli STESSI
# commenti. Riceve due serie booleane (True = risposta giusta) con gli stessi indici e
# conta i commenti dove i due si dividono: solo_a (giusto solo A) e solo_b (giusto solo B).
# Se non ci fosse nessuna differenza reale, i due numeri sarebbero circa uguali: il test
# calcola la probabilita' (p-value) di vedere una divisione cosi' o piu' sbilanciata per
# puro caso. Convenzione: p < 0.05 = differenza probabilmente reale. Con pochi commenti
# serve una differenza molto grande per arrivarci. Restituisce (solo_a, solo_b, p_value).
def mcnemar_exact(correct_a, correct_b):
    correct_a, correct_b = correct_a.astype(bool), correct_b.astype(bool)
    only_a = int((correct_a & ~correct_b).sum())
    only_b = int((~correct_a & correct_b).sum())
    discordant = only_a + only_b
    if discordant == 0:
        return only_a, only_b, 1.0
    return only_a, only_b, float(binomtest(min(only_a, only_b), discordant, 0.5).pvalue)


# Per ogni modello risponde alla domanda: "tradurre le emoji in parole cambia i risultati?"
# Confronta le risposte sulle due versioni del testo con il test di McNemar. Colonne:
#   n                      commenti confrontati
#   acc_<versione>         accuracy con ciascuna versione
#   solo_<versione>        commenti giusti SOLO con quella versione
#   p_value                probabilita' che la differenza sia dovuta al caso
def compare_versions(merged, version_a=VERSION_CLEAN, version_b=VERSION_EMOJI):
    rows = []
    for model, g in merged.groupby("model", sort=False):
        wide = (g.drop_duplicates(["comment_id", "version"])
                 .pivot(index="comment_id", columns="version", values="correct"))
        if version_a not in wide.columns or version_b not in wide.columns:
            continue
        pair = wide[[version_a, version_b]].dropna().astype(bool)
        only_a, only_b, p_value = mcnemar_exact(pair[version_a], pair[version_b])
        rows.append({
            "model": model,
            "n": len(pair),
            f"acc_{version_a}": float(pair[version_a].mean()),
            f"acc_{version_b}": float(pair[version_b].mean()),
            f"solo_{version_a}": only_a,
            f"solo_{version_b}": only_b,
            "p_value": p_value,
        })
    return pd.DataFrame(rows)


# Per una versione del testo confronta i modelli a due a due con il test di McNemar:
# "il modello A e' davvero migliore di B, o la differenza e' dovuta al caso?". Con 4 modelli
# ci sono 6 coppie; piu' confronti fai, piu' e' facile trovare una differenza "significativa"
# per caso, quindi p_bonferroni moltiplica ogni p_value per il numero di coppie (massimo 1):
# e' il valore da guardare. Se c'e' un solo modello restituisce una tabella vuota.
def compare_models(merged, version):
    columns = ["version", "model_a", "model_b", "n", "solo_a_giusto", "solo_b_giusto",
               "p_value", "p_bonferroni"]
    sub = merged[merged["version"] == version]
    models = list(dict.fromkeys(sub["model"]))
    if len(models) < 2:
        return pd.DataFrame(columns=columns)

    wide = (sub.drop_duplicates(["comment_id", "model"])
               .pivot(index="comment_id", columns="model", values="correct"))
    pairs = list(combinations(models, 2))
    rows = []
    for model_a, model_b in pairs:
        pair = wide[[model_a, model_b]].dropna().astype(bool)
        only_a, only_b, p_value = mcnemar_exact(pair[model_a], pair[model_b])
        rows.append({
            "version": version, "model_a": model_a, "model_b": model_b, "n": len(pair),
            "solo_a_giusto": only_a, "solo_b_giusto": only_b,
            "p_value": p_value, "p_bonferroni": min(1.0, p_value * len(pairs)),
        })
    return pd.DataFrame(rows, columns=columns)


# Estrae `n` commenti sbagliati da un modello su una versione del testo, scelti a caso (con
# seme fisso, quindi ripetibili), con la tua etichetta e quella del modello. Serve per
# l'analisi degli errori: leggendoli si vede cosa va storto (sarcasmo, negazioni, emoji,
# slang...). L'ultima colonna e' il testo ESATTO che il modello ha ricevuto (text_clean o
# text_emoji, a seconda della versione). Contiene solo testo gia' ripulito e utenti
# pseudonimizzati.
def error_examples(merged, model, version, n=10, seed=42):
    wrong = merged[(merged["model"] == model) & (merged["version"] == version) & ~merged["correct"]]
    text_column = version if version in wrong.columns else "text_clean"
    columns = [c for c in ("comment_id", "user", "label_true", "label_pred", text_column) if c in wrong.columns]
    return wrong.sample(n=min(n, len(wrong)), random_state=seed)[columns].reset_index(drop=True)


# Grafico a barre di una metrica (default macro_f1): un gruppo per modello, una barra per
# versione del testo. Riceve la tabella di compute_metrics() per UN solo sottoinsieme
# (es. metrics[metrics["subset"] == "tutti"]). Restituisce la figura matplotlib: dal
# notebook si vede da sola, da codice si salva con fig.savefig("nome.png").
def plot_metric(metrics, metric="macro_f1", title=None):
    import matplotlib.pyplot as plt

    order = list(dict.fromkeys(metrics["model"]))
    table = metrics.pivot(index="model", columns="version", values=metric).reindex(order)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    table.plot(kind="bar", ax=ax, rot=0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} per modello e versione del testo")
    ax.legend(title="versione del testo")
    fig.tight_layout()
    return fig


# Disegna le matrici di confusione di confusion_matrices() una accanto all'altra (`ncols`
# per riga), con il numero di commenti scritto in ogni casella: righe = etichetta vera,
# colonne = predizione. Restituisce la figura matplotlib.
def plot_confusions(matrices, ncols=4):
    import matplotlib.pyplot as plt

    nrows = -(-len(matrices) // ncols)     # divisione per eccesso
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows), squeeze=False)
    flat = axes.ravel()

    for ax, ((model, version), matrix) in zip(flat, matrices.items()):
        ax.imshow(matrix.values, cmap="Blues")
        top = matrix.values.max() or 1
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = int(matrix.values[i, j])
                ax.text(j, i, value, ha="center", va="center",
                        color="white" if value > top / 2 else "black")
        ax.set_xticks(range(len(LABELS)), labels=LABELS, rotation=45, ha="right")
        ax.set_yticks(range(len(LABELS)), labels=LABELS)
        ax.set_xlabel("predetto")
        ax.set_ylabel("vero")
        ax.set_title(f"{model}\n{version}", fontsize=9)

    for ax in flat[len(matrices):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


# Esegue tutta la valutazione e restituisce un dizionario con i risultati:
#   merged      tabella lunga (vedi merge_predictions)
#   metrics     metriche per sottoinsieme, modello e versione
#   confusions  matrici di confusione
#   versions    McNemar: emoji grezze contro emoji tradotte, per ogni modello
#   models      McNemar: modelli a confronto, per ogni versione del testo
def run_evaluation(labeled, predictions):
    merged = merge_predictions(labeled, predictions)
    versions = list(dict.fromkeys(merged["version"]))
    return {
        "merged": merged,
        "metrics": metrics_by_subset(merged),
        "confusions": confusion_matrices(merged),
        "versions": compare_versions(merged),
        "models": pd.concat([compare_models(merged, v) for v in versions], ignore_index=True),
    }


# Salva in out_dir (default data/results/) i risultati di run_evaluation():
#   metriche.csv, mcnemar_versioni.csv, mcnemar_modelli.csv, confusioni.csv  (tabelle)
#   macro_f1.png, matrici_confusione.png                                     (grafici)
# Il grafico delle barre riguarda tutti i commenti. I file sovrascrivono quelli precedenti.
def save_results(results, out_dir=RESULTS_DIR):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results["metrics"].to_csv(out / "metriche.csv", index=False)
    results["versions"].to_csv(out / "mcnemar_versioni.csv", index=False)
    results["models"].to_csv(out / "mcnemar_modelli.csv", index=False)

    confusion_rows = []
    for (model, version), matrix in results["confusions"].items():
        for true_row in matrix.index:
            for pred_col in matrix.columns:
                confusion_rows.append({
                    "model": model, "version": version,
                    "vero": true_row.removeprefix("vero_"),
                    "predetto": pred_col.removeprefix("pred_"),
                    "n": int(matrix.loc[true_row, pred_col]),
                })
    pd.DataFrame(confusion_rows).to_csv(out / "confusioni.csv", index=False)

    everyone = results["metrics"][results["metrics"]["subset"] == "tutti"]
    plot_metric(everyone, "macro_f1", "Macro-F1 per modello e versione del testo (tutti i commenti)") \
        .savefig(out / "macro_f1.png", dpi=150)
    plot_confusions(results["confusions"]).savefig(out / "matrici_confusione.png", dpi=150)
    print(f"\nRisultati salvati in {out}")


# Stampa a video il rapporto completo: riassunto dei commenti, metriche per sottoinsieme,
# McNemar tra versioni e tra modelli, e (se n_errors > 0) esempi di commenti sbagliati.
def _print_report(results, n_errors=0):
    merged = results["merged"]
    comments = merged.drop_duplicates("comment_id")
    counts = comments["label_true"].value_counts()
    print("Commenti valutati: {} | {} | con emoji: {}".format(
        len(comments),
        ", ".join(f"{label}: {counts.get(label, 0)}" for label in LABELS),
        int(comments["has_emoji"].sum())))

    for subset, table in results["metrics"].groupby("subset", sort=False):
        n_comments = int(table["n"].max())
        print(f"\n== Metriche: {subset} (n = {n_comments}) ==")
        if n_comments < MIN_RELIABLE:
            print(f"   attenzione: meno di {MIN_RELIABLE} commenti, risultati poco affidabili")
        print(table.drop(columns="subset").round(3).to_string(index=False))

    no_neutral = [m for m, g in merged.groupby("model", sort=False) if not (g["label_pred"] == "neutro").any()]
    if no_neutral and (comments["label_true"] == "neutro").any():
        print(f"\nNota: {', '.join(no_neutral)} non predice mai 'neutro': per un confronto equo "
              f"guarda il sottoinsieme 'senza_neutro'.")

    if not results["versions"].empty:
        print("\n== Emoji grezze o tradotte? McNemar per ogni modello (p < 0.05: differenza probabilmente reale) ==")
        print(results["versions"].round(3).to_string(index=False))

    if not results["models"].empty:
        print("\n== Modelli a confronto, due a due (guarda p_bonferroni) ==")
        print(results["models"].round(3).to_string(index=False))

    if n_errors > 0:
        for (model, version) in results["confusions"]:
            examples = error_examples(merged, model, version, n=n_errors)
            print(f"\n== Errori di {model} su {version} ({len(examples)} esempi) ==")
            print(examples.to_string(index=False, max_colwidth=90) if len(examples) else "nessun errore")


# Punto di ingresso: viene eseguito solo con `python src/evaluate.py ...`. Carica
# dataset etichettato e predizioni, calcola tutto e stampa il rapporto. Se un file manca o
# le etichette sono incomplete si ferma con un messaggio che dice cosa correggere.
def main():
    parser = argparse.ArgumentParser(
        description="Valuta i modelli: predizioni contro le etichette scritte a mano.",
        epilog="""esempi:
  python src/evaluate.py                    tutte le tabelle a video
  python src/evaluate.py --errors 5         + 5 commenti sbagliati per ogni modello e versione
  python src/evaluate.py --save             + CSV e grafici in data/results/""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--predictions", default=str(PREDICTIONS_PATH),
                        help="CSV delle predizioni (default: %(default)s)")
    parser.add_argument("--labels", default=None,
                        help="CSV con le etichette scritte a mano (default: data/labeled/etichettati_a_mano.csv)")
    parser.add_argument("--errors", type=int, default=0, metavar="N",
                        help="stampa N commenti sbagliati per ogni modello e versione (default 0 = nessuno)")
    parser.add_argument("--save", action="store_true",
                        help="salva tabelle (CSV) e grafici (PNG) in --out-dir")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR),
                        help="dove salvare con --save (default: %(default)s)")
    args = parser.parse_args()

    try:
        labeled = load_labeled_dataset(args.labels)
        predictions = load_predictions(args.predictions)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Impossibile partire: {exc}")

    results = run_evaluation(labeled, predictions)
    _print_report(results, args.errors)

    if args.save:
        import matplotlib
        matplotlib.use("Agg")        # grafici solo su file: nessuna finestra
        save_results(results, args.out_dir)


if __name__ == "__main__":
    main()