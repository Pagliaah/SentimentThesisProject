# =====================================================================
# preprocessing.py - dai commenti grezzi al dataset pronto per i modelli
#
# Flusso (nello stesso ordine in cui sono scritte le funzioni):
#   1. load_raw_comments()    legge i CSV di data/raw/
#   2. prepare_comments()     pulisce, toglie i doppioni, sostituisce gli username
#                             con pseudonimi, crea le due versioni del testo
#                             (emoji grezze / emoji tradotte in parole)
#   3. create_label_set()     estrae a caso i commenti da etichettare a mano e li
#                             salva in un "set" numerato (etichettati_a_mano_00.csv,
#                             _01.csv, ...): puoi crearne quanti ne vuoi, uno alla
#                             volta, e i commenti non si ripetono mai da un set all'altro
#   4. attach_labels()        riunisce le tue etichette (di uno o piu' set) al dataset
#
# Uso da terminale (crea il PROSSIMO set, senza sovrascrivere quelli gia' fatti):
#   python src/preprocessing.py --n 300
# =====================================================================

import argparse
import hashlib
import re
from pathlib import Path

import emoji
import pandas as pd


# ---------------------------------------------------------------------
# PERCORSI ed ETICHETTE
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
LABELED_DIR = ROOT_DIR / "data" / "labeled"

# I set di etichette si chiamano etichettati_a_mano_00.csv, etichettati_a_mano_01.csv...
LABEL_SET_PREFIX = "etichettati_a_mano"
SET_FILE_RE = re.compile(rf"^{LABEL_SET_PREFIX}_(\d+)\.csv$")
# Nomi usati in passato per un unico file di etichette (prima dei set numerati).
LEGACY_LABEL_FILES = (f"{LABEL_SET_PREFIX}.csv", "da_etichettare.csv")

VALID_LABELS = {"positivo", "negativo", "neutro"}


# ---------------------------------------------------------------------
# ESPRESSIONI REGOLARI
# ---------------------------------------------------------------------
# indirizzo web; la punteggiatura finale ("!" "." ")") resta fuori, perche' fa parte della frase
URL_RE = re.compile(r"(?:https?://|www\.)\S*[^\s.,;:!?)\]\"']")
# @username: lettere, numeri, punti e underscore; il punto finale non fa parte del nome
MENTION_RE = re.compile(r"@[A-Za-z0-9_](?:[A-Za-z0-9._]*[A-Za-z0-9_])?")
SPACES_RE = re.compile(r"\s+")
SKIN_TONE_RE = re.compile("[\U0001F3FB-\U0001F3FF]")     # modificatori del colore della pelle
USERNAME_RE = re.compile(r"[A-Za-z0-9._]+")              # username Instagram valido

# Caratteri "inventati" (area privata di Unicode) usati come delimitatori quando la
# libreria emoji traduce le emoji: non compaiono mai nei commenti veri, quindi si
# riconoscono con certezza i pezzi tradotti senza toccare i ":" o "_" del testo normale.
EMOJI_OPEN, EMOJI_CLOSE = "\ue000", "\ue001"
EMOJI_TOKEN_RE = re.compile(EMOJI_OPEN + "(.*?)" + EMOJI_CLOSE)


# Legge tutti i CSV di commenti in data/raw/ (quelli creati da scraping.py, uno per
# profilo) e li unisce in un unico DataFrame. `pattern` sceglie quali file leggere
# (di default tutti i commenti_*.csv). Se non ne trova nessuno si ferma con un errore
# chiaro, invece di restituire una tabella vuota.
def load_raw_comments(pattern="commenti_*.csv"):
    files = sorted(RAW_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Nessun file '{pattern}' in {RAW_DIR}")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


# Pulizia minima di un commento. NON toglie emoji, punteggiatura o maiuscole: sono
# informazioni che i modelli possono usare. Fa solo tre cose:
#   - gli indirizzi web diventano "http";
#   - le menzioni (@nome) diventano "@user": cosi' nel dataset non restano nomi di
#     persone vere, ed e' anche il formato che i modelli addestrati sui tweet si
#     aspettano;
#   - a capo e spazi multipli diventano un solo spazio.
def basic_clean(text):
    text = str(text)
    text = URL_RE.sub("http", text)
    text = MENTION_RE.sub("@user", text)
    return SPACES_RE.sub(" ", text).strip()


# Passo interno: toglie i colori della pelle e fa tradurre le emoji alla libreria
# `emoji` (in italiano), racchiudendo ogni nome tra i delimitatori speciali.
# Esempio: "Bravo 👏" -> "Bravo \ue000mani_che_applaudono\ue001".
# I colori della pelle (🏻 🏼 ...) vengono tolti prima perche' tradotti diventerebbero
# parole inutili ("carnagione chiara") che sporcano il testo.
def _mark_emojis(text):
    text = SKIN_TONE_RE.sub("", text)
    return emoji.demojize(text, language="it", delimiters=(EMOJI_OPEN, EMOJI_CLOSE))


# Versione del testo con le emoji tradotte in parole italiane, per i modelli che non
# le capiscono. Esempio: "Bravo 👏" -> "Bravo mani che applaudono". Gli underscore
# dei nomi delle emoji diventano spazi, in modo che il modello veda parole separate.
def translate_emojis(text):
    marked = _mark_emojis(text)
    translated = EMOJI_TOKEN_RE.sub(lambda m: " " + m.group(1).replace("_", " ") + " ", marked)
    return SPACES_RE.sub(" ", translated).strip()


# True se il commento e' fatto SOLO di emoji (es. "👏👏👏"), False altrimenti.
# Serve per l'analisi degli errori: i commenti di sole emoji sono il caso in cui la
# differenza tra "emoji grezze" ed "emoji tradotte" pesa di piu'.
def is_emoji_only(text):
    marked = _mark_emojis(text)
    if EMOJI_OPEN not in marked:
        return False
    rest = EMOJI_TOKEN_RE.sub("", marked)
    return not re.search(r"\w", rest)       # niente lettere/numeri oltre alle emoji


# Crea un identificativo stabile per un commento: i primi 12 caratteri di un hash di
# (post, username, testo). Resta uguale anche se rifai lo scraping, quindi le etichette
# scritte a mano restano collegate al commento giusto. Comincia con "c" cosi' Excel o
# Numbers non lo scambiano per un numero in notazione scientifica (es. "12e4567").
def make_comment_id(url_post, username, text):
    key = f"{url_post}|{username}|{text}".encode("utf-8")
    return "c" + hashlib.sha1(key).hexdigest()[:12]


# Trasforma i commenti grezzi nel dataset usato da tutto il resto. Passi:
#   1. toglie righe incomplete e username non validi (rete di sicurezza per CSV
#      raccolti con versioni vecchie dello scraper);
#   2. toglie i doppioni (stesso post, stesso utente, stesso testo);
#   3. crea comment_id (vedi make_comment_id);
#   4. sostituisce gli username con pseudonimi user_001, user_002... (in ordine
#      alfabetico, quindi uguali a ogni esecuzione sugli stessi dati);
#   5. crea le colonne di testo: text_clean (emoji grezze), text_emoji (emoji
#      tradotte in parole) ed emoji_only;
#   6. scarta i commenti rimasti vuoti.
# Restituisce (dataset, mappa_username). Il dataset NON contiene gli username veri ne'
# il testo originale (potrebbe avere @menzioni), cosi' non finiscono per sbaglio negli
# output del notebook. La mappa serve solo a te: se vuoi conservarla, salvala in data/
# (cartella ignorata da git), mai nel repository.
def prepare_comments(raw):
    df = raw.dropna(subset=["url_post", "username", "text"]).copy()
    df = df[df["username"].astype(str).map(lambda u: bool(USERNAME_RE.fullmatch(u)))]
    df = df.drop_duplicates(subset=["url_post", "username", "text"])

    df["comment_id"] = [make_comment_id(p, u, t)
                        for p, u, t in zip(df["url_post"], df["username"], df["text"])]

    usernames = sorted(df["username"].unique())
    username_map = {name: f"user_{i:03d}" for i, name in enumerate(usernames, start=1)}
    df["user"] = df["username"].map(username_map)

    df["text_clean"] = df["text"].map(basic_clean)
    df["text_emoji"] = df["text_clean"].map(translate_emojis)
    df["emoji_only"] = df["text_clean"].map(is_emoji_only)
    df = df[df["text_clean"] != ""]

    columns = ["comment_id", "url_post", "user", "text_clean", "text_emoji", "emoji_only", "likes"]
    return df[columns].reset_index(drop=True), username_map


# ---------------------------------------------------------------------
# SET DI ETICHETTE
# Ogni set e' un file data/labeled/etichettati_a_mano_NN.csv (NN = 00, 01, 02...) con tre
# colonne: comment_id, text_clean, label_true. Si possono creare piu' set in momenti
# diversi (per esempio 300 commenti oggi e altri 150 la settimana prossima): i commenti
# di un set non compaiono mai in un altro, e per valutare i modelli si possono usare
# tutti i set insieme o solo alcuni.
# ---------------------------------------------------------------------

# Percorso del file di un set: label_set_path(1) o label_set_path("01") ->
# data/labeled/etichettati_a_mano_01.csv. Un numero non valido da' un errore chiaro.
def label_set_path(number):
    try:
        number = int(number)
    except (TypeError, ValueError):
        raise ValueError(f"'{number}' non e' un numero di set valido (esempi: 00, 01, 02)")
    return LABELED_DIR / f"{LABEL_SET_PREFIX}_{number:02d}.csv"


# Elenca i set esistenti, in ordine di numero, come lista di coppie (numero, percorso),
# per esempio [("00", Path(...)), ("01", Path(...))]. Lista vuota se non ce ne sono.
def list_label_sets():
    found = {}
    for path in LABELED_DIR.glob(f"{LABEL_SET_PREFIX}_*.csv"):
        match = SET_FILE_RE.match(path.name)
        if match:
            found[f"{int(match.group(1)):02d}"] = path
    return sorted(found.items(), key=lambda item: int(item[0]))


# Numero del prossimo set libero: 0 se non ce ne sono, altrimenti l'ultimo + 1.
def next_set_number():
    sets = list_label_sets()
    return int(sets[-1][0]) + 1 if sets else 0


# Trasforma i numeri di set scelti dall'utente (es. ["00", "02"]) nei percorsi dei file.
# Senza numeri restituisce TUTTI i set esistenti. Se non c'e' nessun set, o ne viene
# chiesto uno che non esiste, si ferma con un messaggio che dice cosa fare. Se trova un
# vecchio file di etichette senza numero (etichettati_a_mano.csv / da_etichettare.csv)
# suggerisce di rinominarlo come set 00.
def resolve_label_sets(numbers=None):
    available = list_label_sets()
    if not available:
        legacy = [name for name in LEGACY_LABEL_FILES if (LABELED_DIR / name).exists()]
        hint = (f" Trovato {legacy[0]}: rinominalo in {label_set_path(0).name} per usarlo come set 00."
                if legacy else "")
        raise FileNotFoundError(f"Nessun set di etichette in {LABELED_DIR}. "
                                f"Creane uno con: python src/preprocessing.py.{hint}")
    if not numbers:
        return [path for _, path in available]

    by_number = dict(available)
    chosen = []
    for number in numbers:
        key = f"{int(label_set_path(number).stem.rsplit('_', 1)[1]):02d}"
        if key not in by_number:
            raise FileNotFoundError(f"Il set {key} non esiste. Set disponibili: {', '.join(by_number)}")
        if by_number[key] not in chosen:
            chosen.append(by_number[key])
    return chosen


# Legge un file di etichette. Riconosce da solo il separatore e la codifica con o senza BOM. 
# Tutto come testo.
def _read_labels_csv(path):
    return pd.read_csv(path, sep=None, engine="python", dtype=str, encoding="utf-8-sig")


# Nome del set a cui appartiene un file ("00", "01"...); se il nome non segue lo schema
# usa il nome del file senza estensione.
def _set_name(path):
    match = SET_FILE_RE.match(Path(path).name)
    return f"{int(match.group(1)):02d}" if match else Path(path).stem


# Identificativi di tutti i commenti gia' presenti in un set (etichettati o ancora da
# etichettare): servono a non estrarli una seconda volta.
def used_comment_ids():
    ids = set()
    for _, path in list_label_sets():
        ids.update(_read_labels_csv(path)["comment_id"].dropna())
    return ids


# Estrae `n` commenti a caso da etichettare a mano e restituisce una tabella con tre
# colonne: comment_id, text_clean (con le emoji vere, cosi' le leggi come le leggerebbe
# una persona) e label_true, vuota. Il campionamento e' casuale ma ripetibile: con lo
# stesso `seed` e gli stessi dati ottieni sempre lo stesso campione. I commenti in
# `exclude_ids` (quelli gia' in altri set) non vengono mai estratti. Se i commenti
# disponibili sono meno di `n` li prende tutti.
def sample_for_labeling(df, n=300, seed=42, exclude_ids=None):
    pool = df if not exclude_ids else df[~df["comment_id"].isin(exclude_ids)]
    sample = pool.sample(n=min(n, len(pool)), random_state=seed)
    return pd.DataFrame({
        "comment_id": sample["comment_id"].values,
        "text_clean": sample["text_clean"].values,
        "label_true": "",
    })


# Crea un nuovo set da etichettare e lo salva in data/labeled/. `number` e' il numero
# del set (es. 3 oppure "03"); se omesso usa il primo libero (00, poi 01...). Estrae
# `n` commenti a caso tra quelli che NON sono gia' in un altro set. Non sovrascrive
# mai: se il file esiste solleva FileExistsError; se non resta nessun commento libero
# solleva ValueError (serve altro scraping). Restituisce (percorso, campione).
# Il file e' salvato in UTF-8 con BOM, che permette a Excel e Numbers di mostrare bene
# emoji e lettere accentate.
def create_label_set(dataset, n=300, seed=42, number=None):
    number = next_set_number() if number is None else number
    path = label_set_path(number)
    if path.exists():
        raise FileExistsError(f"{path.name} esiste gia': non lo sovrascrivo.")

    sample = sample_for_labeling(dataset, n, seed, exclude_ids=used_comment_ids())
    if sample.empty:
        raise ValueError("Tutti i commenti sono gia' in un set di etichette: serve altro scraping.")

    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False, encoding="utf-8-sig")
    return path, sample


# Legge le etichette che hai scritto a mano e le riunisce al dataset, aggiungendo le
# colonne label_true e set (il numero del set di provenienza). `labeled_csv` puo' essere
# un solo file o una lista di file (vedi resolve_label_sets). Controlla che il lavoro a
# mano sia a posto: se qualche riga non ha etichetta o ne ha una diversa da
# positivo/negativo/neutro, si ferma e dice in quale file e a quali righe (numerate come
# le vedi nel foglio di calcolo, intestazione = riga 1). Si ferma anche se uno stesso
# commento compare in piu' righe o in piu' set. Ignora maiuscole e spazi. Restituisce
# solo i commenti etichettati.
def attach_labels(df, labeled_csv):
    paths = list(labeled_csv) if isinstance(labeled_csv, (list, tuple)) else [labeled_csv]

    frames = []
    for path in paths:
        part = _read_labels_csv(path)
        missing_columns = {"comment_id", "label_true"} - set(part.columns)
        if missing_columns:
            raise ValueError(f"{Path(path).name}: mancano le colonne {sorted(missing_columns)}")
        part["label_true"] = part["label_true"].str.strip().str.lower()
        part["set"] = _set_name(path)
        part["file"] = Path(path).name
        part["riga"] = part.index + 2
        frames.append(part)
    labels = pd.concat(frames, ignore_index=True)

    missing = labels["label_true"].isna() | (labels["label_true"] == "")
    invalid = ~missing & ~labels["label_true"].isin(VALID_LABELS)
    problems = []
    for name in labels["file"].unique():
        in_file = labels["file"] == name
        if (missing & in_file).any():
            problems.append(f"{name}: senza etichetta alle righe {labels.loc[missing & in_file, 'riga'].tolist()}")
        if (invalid & in_file).any():
            problems.append(f"{name}: etichetta non valida alle righe "
                            f"{labels.loc[invalid & in_file, 'riga'].tolist()} "
                            f"(valide: {', '.join(sorted(VALID_LABELS))})")
    if problems:
        raise ValueError("Etichette da correggere -> " + "; ".join(problems))

    repeated = labels["comment_id"].duplicated(keep=False)
    if repeated.any():
        sets = ", ".join(sorted(labels.loc[repeated, "set"].unique()))
        raise ValueError(f"{labels.loc[repeated, 'comment_id'].nunique()} commenti compaiono piu' volte "
                         f"nei set di etichette ({sets}): ogni commento deve stare in un solo set.")

    merged = df.merge(labels[["comment_id", "label_true", "set"]], on="comment_id", how="inner")
    if len(merged) < len(labels):
        print(f"Attenzione: {len(labels) - len(merged)} commenti etichettati non sono nel dataset "
              f"(scraping rifatto con dati diversi?)")
    return merged


# Punto di ingresso: viene eseguito solo con `python src/preprocessing.py ...`.
# Prepara il dataset e crea un NUOVO set da etichettare in data/labeled/
# (etichettati_a_mano_NN.csv): per default il primo numero libero, oppure quello
# indicato con --set. I commenti estratti sono diversi da quelli degli altri set.
# Non sovrascrive mai un set esistente, cosi' non puoi perdere le etichette scritte
# a mano.
def main():
    parser = argparse.ArgumentParser(description="Crea un set di commenti da etichettare a mano")
    parser.add_argument("--n", type=int, default=300, help="quanti commenti estrarre (default 300)")
    parser.add_argument("--seed", type=int, default=42, help="seme del campionamento casuale")
    parser.add_argument("--set", dest="set_number", default=None, metavar="NN",
                        help="numero del set da creare (es. 03); se omesso, il primo libero")
    args = parser.parse_args()

    dataset, _ = prepare_comments(load_raw_comments())
    print(f"Commenti pronti: {len(dataset)} (di cui solo emoji: {int(dataset['emoji_only'].sum())})")

    existing = [number for number, _ in list_label_sets()]
    if existing:
        print(f"Set di etichette gia' presenti: {', '.join(existing)}")

    try:
        path, sample = create_label_set(dataset, args.n, args.seed, args.set_number)
    except FileExistsError as exc:
        print(exc)
        return
    except ValueError as exc:
        raise SystemExit(str(exc))

    if len(sample) < args.n:
        print(f"Attenzione: restavano solo {len(sample)} commenti non ancora in un set.")
    print(f"Creato {path} con {len(sample)} commenti: compila la colonna label_true "
          f"(positivo / negativo / neutro).")


if __name__ == "__main__":
    main()