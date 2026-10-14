# =====================================================================
# preprocessing.py - dai commenti grezzi al dataset pronto per i modelli
#
# Flusso (nello stesso ordine in cui sono scritte le funzioni):
#   1. load_raw_comments()    legge i CSV di data/raw/
#   2. prepare_comments()     pulisce, toglie i doppioni, sostituisce gli username
#                             con pseudonimi, crea le due versioni del testo
#                             (emoji grezze / emoji tradotte in parole)
#   3. sample_for_labeling()  estrae a caso i commenti da etichettare a mano
#   4. attach_labels()        riunisce le tue etichette al dataset
#
# Uso da terminale (crea il file da etichettare, senza sovrascrivere quello gia' fatto):
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
TO_LABEL_FILE = LABELED_DIR / "da_etichettare.csv"

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


# Estrae `n` commenti a caso da etichettare a mano e restituisce una tabella con tre
# colonne: comment_id, text_clean (con le emoji vere, cosi' le leggi come le leggerebbe
# una persona) e label_true, vuota. Il campionamento e' casuale ma ripetibile: con lo
# stesso `seed` ottieni sempre lo stesso campione. Se i commenti sono meno di `n`
# li prende tutti.
def sample_for_labeling(df, n=300, seed=42):
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    return pd.DataFrame({
        "comment_id": sample["comment_id"].values,
        "text_clean": sample["text_clean"].values,
        "label_true": "",
    })


# Legge il file che hai etichettato a mano e lo riunisce al dataset, aggiungendo la
# colonna label_true. Controlla che il lavoro a mano sia a posto: se qualche riga non
# ha etichetta o ne ha una diversa da positivo/negativo/neutro, si ferma e ti dice
# quali righe correggere (numerate come le vedi nel foglio di calcolo, intestazione =
# riga 1). Legge sia CSV con la virgola sia con il punto e virgola (Excel italiano) e
# ignora maiuscole e spazi. Restituisce solo i commenti etichettati.
def attach_labels(df, labeled_csv):
    labels = pd.read_csv(labeled_csv, sep=None, engine="python", dtype=str, encoding="utf-8-sig")
    labels["label_true"] = labels["label_true"].str.strip().str.lower()

    missing = labels["label_true"].isna() | (labels["label_true"] == "")
    invalid = ~missing & ~labels["label_true"].isin(VALID_LABELS)
    if missing.any() or invalid.any():
        parts = []
        if missing.any():
            parts.append(f"senza etichetta: righe {list((labels.index[missing] + 2))}")
        if invalid.any():
            parts.append(f"etichetta non valida: righe {list((labels.index[invalid] + 2))} "
                         f"(valide: {', '.join(sorted(VALID_LABELS))})")
        raise ValueError("Etichette da correggere -> " + "; ".join(parts))

    merged = df.merge(labels[["comment_id", "label_true"]], on="comment_id", how="inner")
    if len(merged) < len(labels):
        print(f"Attenzione: {len(labels) - len(merged)} commenti etichettati non sono nel dataset "
              f"(scraping rifatto con dati diversi?)")
    return merged


# Punto di ingresso: viene eseguito solo con `python src/preprocessing.py`.
# Prepara il dataset e crea data/labeled/da_etichettare.csv con il campione da
# etichettare. Se quel file esiste gia' NON lo tocca, cosi' non puoi perdere per
# sbaglio le etichette scritte a mano. Il file e' salvato in UTF-8 con BOM, che
# permette a Excel e Numbers di mostrare bene emoji e lettere accentate.
def main():
    parser = argparse.ArgumentParser(description="Prepara il campione di commenti da etichettare")
    parser.add_argument("--n", type=int, default=300, help="quanti commenti estrarre (default 300)")
    parser.add_argument("--seed", type=int, default=42, help="seme del campionamento casuale")
    args = parser.parse_args()

    dataset, _ = prepare_comments(load_raw_comments())
    print(f"Commenti pronti: {len(dataset)} (di cui solo emoji: {int(dataset['emoji_only'].sum())})")

    if TO_LABEL_FILE.exists():
        print(f"{TO_LABEL_FILE.name} esiste gia': non lo sovrascrivo.")
        return

    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    sample = sample_for_labeling(dataset, args.n, args.seed)
    sample.to_csv(TO_LABEL_FILE, index=False, encoding="utf-8-sig")
    print(f"Creato {TO_LABEL_FILE} con {len(sample)} commenti: compila la colonna label_true "
          f"(positivo / negativo / neutro).")


if __name__ == "__main__":
    main()