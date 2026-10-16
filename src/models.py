# =====================================================================
# Quattro modelli di sentiment analysis (tutti gratuiti, eseguiti in locale) con un'interfaccia comune:
# 
#     modello.predict(lista_di_testi) -> lista di etichette
#                                        ("positivo" | "negativo" | "neutro")
# 
# Modelli:
#     1. feel-it                (Hugging Face, italiano, solo positivo/negativo)
#     2. neuraly BERT italiano  (Hugging Face, italiano, 3 classi)
#     3. cardiffnlp XLM-R       (Hugging Face, multilingua, 3 classi)
#     4. nlptown BERT           (Hugging Face, multilingua, 1-5 stelle -> 3 classi)
# 
# Ogni modello gira su ENTRAMBE le versioni del testo (text_clean e text_emoji)
# tramite run_all(), che produce una tabella "lunga" di predizioni salvata su CSV.
#
# Uso da terminale (dalla cartella principale del progetto, con il venv attivo):
#     python src/models.py                       prova rapida: 5 frasi fisse, tutti i modelli
#     python src/models.py f x                   prova rapida con solo feel_it e xlmr_twitter
#     python src/models.py --text "Bravo!"       prova rapida con una frase tua
#     python src/models.py --dataset             tutti i modelli sul dataset etichettato -> CSV
# La prova rapida NON crea nessun file. Tutte le opzioni: python src/models.py --help
# =====================================================================

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import torch

# Etichette comuni a tutto il progetto (le stesse usate in attach_labels).
LABELS = ("positivo", "negativo", "neutro")

# Cartella principale del progetto e file di default per le predizioni. Il percorso parte
# da dove si trova questo file, quindi e' lo stesso sia da terminale sia dal notebook
# (con un percorso relativo, dal notebook il CSV finiva in notebooks/data/...).
ROOT_DIR = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = ROOT_DIR / "data" / "predictions" / "predizioni.csv"


# ---------------------------------------------------------------------------
# Interfaccia comune
# ---------------------------------------------------------------------------
class SentimentModel(ABC):
    """Classe base: ogni modello espone un nome e un metodo predict()."""

    name: str = "modello"

    # Riceve una lista di testi e restituisce una lista di etichette della
    # stessa lunghezza, ciascuna in LABELS. Le sottoclassi lo implementano.
    @abstractmethod
    def predict(self, texts: list[str]) -> list[str]:
        ...

    # Libera la memoria occupata dal modello. Di default non fa nulla;
    # i modelli Hugging Face lo ridefiniscono.
    def unload(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Modelli Hugging Face
# ---------------------------------------------------------------------------
class HFSentimentModel(SentimentModel):
    """Classificatore di sentiment Hugging Face (modello + tokenizer)."""

    # Salva la configurazione senza caricare nulla (il caricamento pesante
    # avviene alla prima predict(), così creare i 4 oggetti è istantaneo).
    # label_map traduce le etichette del modello (config.id2label, in
    # minuscolo) in quelle del progetto. neutral_threshold, se impostato,
    # trasforma in "neutro" le predizioni con confidenza inferiore.
    def __init__(
        self,
        name: str,
        hf_id: str,
        label_map: dict[str, str],
        batch_size: int = 32,
        max_length: int = 128,
        neutral_threshold: float | None = None,
    ):
        self.name = name
        self.hf_id = hf_id
        self.label_map = label_map
        self.batch_size = batch_size
        self.max_length = max_length
        self.neutral_threshold = neutral_threshold
        self.tokenizer = None
        self.model = None
        self._labels: list[str] = []
        # Su Apple Silicon usa la GPU (mps); altrimenti CPU.
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Scarica (la prima volta) e carica modello E tokenizer dallo STESSO
    # nome Hugging Face (nel vecchio progetto il tokenizer era sbagliato).
    # Poi costruisce la lista _labels: per ogni indice di output del modello,
    # l'etichetta del progetto corrispondente. Se il modello usa nomi
    # generici (LABEL_0, LABEL_1...) assume l'ordine standard, che dipende
    # dal numero di classi: 2 classi = negativo, positivo; 3 classi =
    # negativo, neutro, positivo. Se trova un nome sconosciuto (o un numero
    # di classi diverso) si ferma con un errore chiaro invece di etichettare
    # male in silenzio.
    def _load(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        print(f"[{self.name}] carico {self.hf_id} su {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.hf_id)
        self.model.to(self.device)
        self.model.eval()  # niente dropout: predizioni deterministiche

        generic_orders = {
            2: ["negativo", "positivo"],
            3: ["negativo", "neutro", "positivo"],
        }
        generic_order = generic_orders.get(self.model.config.num_labels, [])
        self._labels = []
        for i in range(self.model.config.num_labels):
            raw = str(self.model.config.id2label[i]).lower()
            if raw in self.label_map:
                self._labels.append(self.label_map[raw])
            elif raw.startswith("label_") and i < len(generic_order):
                self._labels.append(generic_order[i])
            else:
                raise ValueError(
                    f"[{self.name}] etichetta del modello non riconosciuta: "
                    f"'{raw}'. Aggiornare label_map."
                )

    # Classifica i testi a blocchi (batch_size alla volta). Per ogni blocco:
    # tokenizza con padding e troncamento a max_length, calcola i logits
    # senza gradienti, li trasforma in probabilità con softmax e prende la
    # classe più probabile. Testi mancanti (NaN) diventano stringa vuota.
    def predict(self, texts: list[str]) -> list[str]:
        self._load()
        texts = ["" if pd.isna(t) else str(t) for t in texts]
        out: list[str] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            confidence, idx = probs.max(dim=-1)

            for conf, j in zip(confidence.tolist(), idx.tolist()):
                label = self._labels[j]
                if self.neutral_threshold is not None and conf < self.neutral_threshold:
                    label = "neutro"
                out.append(label)

        return out

    # Rilascia modello e tokenizer e svuota la cache della GPU: utile per
    # non tenere in memoria tutti i modelli insieme mentre si esegue run_all().
    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        if self.device == "mps":
            torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Registro dei modelli
# ---------------------------------------------------------------------------
# Costruisce i 4 modelli del confronto. Nessun download avviene qui: i pesi
# vengono scaricati alla prima predict(). Gli id Hugging Face vanno
# verificati (esistenza e nomi delle etichette) al primo utilizzo.
def build_models(neutral_threshold: float | None = None) -> list[SentimentModel]:
    three = {"negative": "negativo", "neutral": "neutro", "positive": "positivo"}

    return [
        # Italiano, addestrato su tweet: solo positivo/negativo (niente neutro).
        HFSentimentModel(
            name="feel_it",
            hf_id="MilaNLProc/feel-it-italian-sentiment",
            label_map={"negative": "negativo", "positive": "positivo"},
            neutral_threshold=neutral_threshold,
        ),
        # Italiano, BERT fine-tuned sul sentiment, 3 classi.
        HFSentimentModel(
            name="bert_italiano",
            hf_id="neuraly/bert-base-italian-cased-sentiment",
            label_map=three,
        ),
        # Multilingua, addestrato su tweet, 3 classi.
        HFSentimentModel(
            name="xlmr_twitter",
            hf_id="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            label_map=three,
        ),
        # Multilingua, recensioni 1-5 stelle: 1-2 negativo, 3 neutro, 4-5 positivo.
        HFSentimentModel(
            name="nlptown_stelle",
            hf_id="nlptown/bert-base-multilingual-uncased-sentiment",
            label_map={
                "1 star": "negativo",
                "2 stars": "negativo",
                "3 stars": "neutro",
                "4 stars": "positivo",
                "5 stars": "positivo",
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Esecuzione dell'esperimento
# ---------------------------------------------------------------------------
# Esegue ogni modello su ogni versione del testo e restituisce una tabella
# lunga con colonne: comment_id, model, version, label_pred.
# df deve avere comment_id e le colonne indicate in `versions` (per avere
# text_emoji basta unire il campione etichettato al dataset di
# prepare_comments tramite comment_id). Dopo ogni modello salva il CSV
# (se out_path non è None) e libera la memoria, così un errore a metà
# non fa perdere i risultati già calcolati. I modelli con errore
# vengono saltati con un avviso.
def run_all(
    df: pd.DataFrame,
    models: list[SentimentModel] | None = None,
    versions: tuple[str, ...] = ("text_clean", "text_emoji"),
    out_path: Path | str | None = PREDICTIONS_PATH,
) -> pd.DataFrame:
    missing = [c for c in ("comment_id", *versions) if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne mancanti nel DataFrame: {missing}")

    models = models if models is not None else build_models()
    rows: list[pd.DataFrame] = []

    for model in models:
        for version in versions:
            try:
                preds = model.predict(df[version].tolist())
            except Exception as exc:  # modello non utilizzabile: si prosegue
                print(f"[{model.name}] SALTATO su {version}: {exc}")
                continue
            rows.append(
                pd.DataFrame(
                    {
                        "comment_id": df["comment_id"].values,
                        "model": model.name,
                        "version": version,
                        "label_pred": preds,
                    }
                )
            )
            print(f"[{model.name}] {version}: fatto")

        model.unload()
        if out_path is not None and rows:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            pd.concat(rows, ignore_index=True).to_csv(out_path, index=False)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["comment_id", "model", "version", "label_pred"]
    )


# ---------------------------------------------------------------------------
# Uso da riga di comando
# ---------------------------------------------------------------------------
# Frasi della prova rapida: sono sempre queste, a meno di --text.
ESEMPI = [
    "Grande Prof, complimenti per il lavoro!",
    "Che schifo, non funziona mai niente",
    "Domani a che ora inizia la lezione?",
    "Bellissimo ❤️👏",
    "🖕",
]


# Sceglie i modelli da usare in base a cio' che hai scritto sulla riga di comando.
# Ogni parola puo' essere il nome intero (feel_it) o solo la sua parte iniziale
# (f, feel, x...): basta che identifichi UN solo modello. Senza parole restituisce
# tutti i modelli. Se una parola non corrisponde a nessun modello, o a piu' di uno,
# il programma si ferma con un messaggio che elenca i nomi validi.
def select_models(models: list[SentimentModel], requested: list[str]) -> list[SentimentModel]:
    if not requested:
        return models

    names = ", ".join(m.name for m in models)
    chosen: list[SentimentModel] = []
    for word in requested:
        matches = [m for m in models if m.name.startswith(word.lower())]
        if not matches:
            raise SystemExit(f"Modello '{word}' sconosciuto. Disponibili: {names}")
        if len(matches) > 1:
            raise SystemExit(f"'{word}' e' ambiguo ({', '.join(m.name for m in matches)}): "
                             f"scrivi qualche lettera in piu'.")
        if matches[0] not in chosen:
            chosen.append(matches[0])
    return chosen


# Prova rapida: classifica le frasi con ogni modello scelto, UNO ALLA VOLTA (carica,
# predice, libera la memoria) e poi stampa i risultati frase per frase, con una riga
# per modello. Serve a controllare che i nomi Hugging Face e le etichette funzionino.
# Se un modello da' errore lo segnala e passa al successivo. NON crea nessun file.
def _smoke_test(models: list[SentimentModel], texts: list[str]) -> None:
    results: dict[str, list[str] | str] = {}
    for model in models:
        try:
            results[model.name] = model.predict(texts)
        except Exception as exc:
            results[model.name] = f"ERRORE: {exc}"
        model.unload()

    for i, text in enumerate(texts):
        print(f"\n{i + 1}. {text}")
        for name, result in results.items():
            shown = result if isinstance(result, str) else result[i]
            print(f"     {name:16s} {shown}")


# Esegue i modelli scelti sul dataset vero, con le due versioni del testo, e salva le
# predizioni in out_path (stesso risultato di run_all dal notebook). Passi:
#   1. legge i commenti di data/raw/ e li prepara (prepare_comments);
#   2. li unisce alle etichette di data/labeled/da_etichettare.csv (attach_labels):
#      si usano solo i commenti che hai etichettato, perche' sono gli unici valutabili;
#   3. lancia run_all(), che salva il CSV dopo ogni modello.
# Se mancano i CSV o le etichette sono incomplete si ferma con il messaggio di errore
# di preprocessing, che indica le righe da correggere.
def _run_dataset(models: list[SentimentModel], out_path: Path | str) -> None:
    try:
        import preprocessing as pp            # lanciato come: python src/models.py
    except ImportError:
        from src import preprocessing as pp   # importato come modulo (es. dal notebook)

    try:
        dataset, _ = pp.prepare_comments(pp.load_raw_comments())
        labeled = pp.attach_labels(dataset, pp.TO_LABEL_FILE)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Impossibile partire: {exc}")

    print(f"Commenti etichettati: {len(labeled)} | modelli: {', '.join(m.name for m in models)}")
    preds = run_all(labeled, models=models, out_path=out_path)
    if preds.empty:
        raise SystemExit("Nessuna predizione prodotta: tutti i modelli hanno dato errore.")

    print(f"\nPredizioni salvate in {out_path}")
    print(preds.groupby(["model", "version"]).size().rename("predizioni").to_string())


# Punto di ingresso: viene eseguito solo con `python src/models.py ...`.
# Senza opzioni fa la prova rapida con le 5 frasi fisse e tutti i modelli, senza creare
# file. Le parole libere scelgono i modelli (iniziale o nome), --text cambia le frasi,
# --dataset passa all'esecuzione vera sul dataset etichettato e salva il CSV.
def main() -> None:
    names = ", ".join(m.name for m in build_models())
    parser = argparse.ArgumentParser(
        description="Modelli di sentiment analysis: prova rapida o esecuzione sul dataset.",
        epilog=f"""modelli disponibili: {names}
(basta l'iniziale: f, b, x, n)

esempi:
  python src/models.py                         prova rapida, 5 frasi, tutti i modelli
  python src/models.py f                       prova rapida con il solo feel_it
  python src/models.py f x --text "Bravo!"     una tua frase, solo feel_it e xlmr_twitter
  python src/models.py --dataset               tutti i modelli sul dataset -> CSV
  python src/models.py n --dataset             solo nlptown_stelle sul dataset -> CSV""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("models", nargs="*",
                        help="modelli da usare: nome intero o iniziale (f, b, x, n); se omesso, tutti")
    parser.add_argument("--text", action="append", metavar="FRASE",
                        help="frase da classificare (si puo' ripetere); se omesso, le 5 frasi fisse")
    parser.add_argument("--dataset", action="store_true",
                        help="esegue i modelli sui commenti etichettati (entrambe le versioni del "
                             "testo) e salva le predizioni su CSV")
    parser.add_argument("--out", default=str(PREDICTIONS_PATH),
                        help="dove salvare le predizioni con --dataset (default: %(default)s)")
    parser.add_argument("--neutral-threshold", type=float, default=None, metavar="SOGLIA",
                        help="feel_it non ha la classe neutro: con questa soglia (es. 0.7) le sue "
                             "predizioni poco sicure diventano 'neutro'")
    args = parser.parse_args()

    if args.dataset and args.text:
        parser.error("--text e --dataset non si usano insieme")

    models = select_models(build_models(neutral_threshold=args.neutral_threshold), args.models)

    if args.dataset:
        _run_dataset(models, args.out)
    else:
        _smoke_test(models, args.text or ESEMPI)


if __name__ == "__main__":
    main()