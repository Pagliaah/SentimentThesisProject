# Uso da terminale degli script in `src/`

Ogni script si può lanciare da solo dal terminale, senza aprire il notebook.

## Prima di cominciare

1. Apri il terminale nella cartella principale del progetto (`SentimentProject`).
2. Attiva l'ambiente virtuale: `source venv/bin/activate` (a inizio riga compare `(venv)`).
3. Se non l'hai ancora fatto, installa le librerie: `pip install -r requirements.txt`.
4. Solo per `scraping.py`: nella cartella principale serve il file `.env` con
   `INSTAGRAM_USER=...` e `INSTAGRAM_PASSWORD=...` (vedi `.env.example`).

Forma generale del comando:

```
python src/NOME_SCRIPT.py [argomenti] [--opzioni]
```

Per vedere le opzioni di uno script: `python src/NOME_SCRIPT.py --help`.

---

## `scraping.py`: raccoglie post e commenti di un profilo

```
python src/scraping.py PROFILO [--posts N] [--comments M] [--reset-session]
```

| Parametro | Obbligatorio | Default | Significato |
|---|---|---|---|
| `PROFILO` | sì | | username del profilo pubblico, senza `@` |
| `--posts N` | no | 10 | quanti post tenere (i più commentati) |
| `--comments M` | no | 50 | massimo di commenti da leggere per ogni post |
| `--reset-session` | no | | cancella i cookie salvati e rifà il login con username e password |

**Esempio 1: prova veloce**

```
python src/scraping.py uniurbit --posts 2 --comments 10
```

**Esempio 2: raccolta più grande**

```
python src/scraping.py uniurbit --posts 20 --comments 200
```

**Cosa produce** (due file in `data/raw/`):

- `posts_PROFILO.csv`: likes, comments, url, caption di ogni post
- `commenti_PROFILO.csv`: url_post, username, text, likes di ogni commento

Rilanciando lo script per lo stesso profilo i due file vengono **sovrascritti**: se vuoi
tenere la raccolta precedente, rinominala prima.

**Da sapere**

- Chrome si apre da solo. Mentre legge i post non muovere il mouse sopra la finestra.
- Al primo avvio fa il login e salva la sessione in `data/session/instagram_cookies.json`;
  dagli avvii successivi riusa i cookie, senza rifare il login. Quel file equivale alla
  password: non va mai caricato su GitHub.
- Se il login non funziona, prova con `--reset-session`. Troppi login ravvicinati
  fanno bloccare l'account da Instagram.

---

## `preprocessing.py`: prepara i dati e crea il file da etichettare

```
python src/preprocessing.py [--n N] [--seed S]
```

| Parametro | Obbligatorio | Default | Significato |
|---|---|---|---|
| `--n N` | no | 300 | quanti commenti estrarre per l'etichettatura a mano |
| `--seed S` | no | 42 | seme del campionamento casuale (stesso seme e stessi dati = stesso campione) |

**Esempio 1: campione di 300 commenti**

```
python src/preprocessing.py
```

**Esempio 2: campione di 500 commenti con un altro seme**

```
python src/preprocessing.py --n 500 --seed 7
```

**Cosa fa**: legge tutti i `data/raw/commenti_*.csv`, li prepara, stampa quanti commenti sono
pronti (e quanti fatti solo di emoji) e crea `data/labeled/da_etichettare.csv` con tre
colonne: `comment_id`, `text_clean`, `label_true`. Compila `label_true` con
`positivo`, `negativo` o `neutro`.

**Attenzione**: se `da_etichettare.csv` esiste già, lo script **non lo sovrascrive**, per
non farti perdere le etichette scritte a mano. Per ottenere un campione nuovo, rinomina o
sposta prima il file vecchio. Se rifai lo scraping i commenti cambiano, quindi cambia anche
il campione.

---

## `models.py`: classifica frasi o commenti con i modelli di sentiment

```
python src/models.py [MODELLI] [--text FRASE] [--dataset] [--out FILE] [--neutral-threshold SOGLIA]
```

| Parametro | Obbligatorio | Default | Significato |
|---|---|---|---|
| `MODELLI` | no | tutti | quali modelli usare: nome intero o solo l'iniziale. `f` = feel_it, `b` = bert_italiano, `x` = xlmr_twitter, `n` = nlptown_stelle. Se ne scrivi più di uno, separali con uno spazio |
| `--text FRASE` | no | le 5 frasi fisse | frase da classificare (si può ripetere più volte) |
| `--dataset` | no | | esegue i modelli sui commenti etichettati, con entrambe le versioni del testo, e salva le predizioni in un CSV |
| `--out FILE` | no | `data/predictions/predizioni.csv` | dove salvare il CSV (solo con `--dataset`) |
| `--neutral-threshold SOGLIA` | no | | feel_it non ha la classe "neutro": con questa soglia (es. `0.7`) le sue predizioni poco sicure diventano "neutro" |

Ci sono due modalità.

**Prova rapida** (quella di default): classifica 5 frasi fisse, o le tue con `--text`, e
stampa il risultato di ogni modello. **Non crea nessun file.**

*Esempio 1: 5 frasi fisse, tutti i modelli*

```
python src/models.py
```

*Esempio 2: due modelli e una frase tua*

```
python src/models.py f x --text "Bravo, complimenti!"
```

**Sul dataset** (`--dataset`): legge i commenti di `data/raw/`, li unisce alle etichette di
`data/labeled/da_etichettare.csv` e classifica i commenti etichettati con ogni modello, sia su
`text_clean` sia su `text_emoji`. Salva un CSV con le colonne `comment_id`, `model`,
`version`, `label_pred`. Il CSV viene aggiornato dopo ogni modello, quindi un errore a metà non
fa perdere quanto già calcolato.

*Esempio 1: tutti i modelli*

```
python src/models.py --dataset
```

*Esempio 2: solo nlptown_stelle, salvando in un altro file*

```
python src/models.py n --dataset --out data/predictions/nlptown.csv
```

**Da sapere**

- Il primo avvio di ogni modello scarica i pesi da Hugging Face (centinaia di MB ciascuno,
  serve internet); dalle volte successive usa la copia salvata sul computer.
- Per `--dataset` servono prima lo scraping, il preprocessing e le etichette complete in
  `da_etichettare.csv`: se manca qualcosa lo script si ferma e dice cosa correggere.
- Rilanciando `--dataset` con lo stesso `--out` il file viene **sovrascritto**.

---

## `evaluate.py`

In arrivo: la sezione si aggiunge qui quando lo script è pronto.

---

## Problemi comuni

| Messaggio | Cosa fare |
|---|---|
| `ModuleNotFoundError` | l'ambiente virtuale non è attivo: `source venv/bin/activate` |
| `Credenziali mancanti: controlla il file .env` | controlla che `.env` esista nella cartella principale e abbia i due valori |
| `Nessun file 'commenti_*.csv'` | lancia prima `scraping.py`: serve almeno un file in `data/raw/` |
| `Interrotto: login non riuscito` | prova `--reset-session`; controlla credenziali o richieste di verifica sul telefono |
| `Modello 'z' sconosciuto` | usa il nome intero o l'iniziale giusta: `f`, `b`, `x`, `n` |
| `Impossibile partire: Etichette da correggere` | completa o correggi le righe indicate in `da_etichettare.csv` (valori validi: `positivo`, `negativo`, `neutro`) |