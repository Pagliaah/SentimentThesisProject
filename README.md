
# SentimentThesisProject
Scraping di commenti Instagram e confronto tra modelli di sentiment analysis, valutati su un set etichettato manualmente. Tesi triennale.

# Uso da terminale degli script in `src/`

Ogni script si può lanciare da solo dal terminale, senza aprire il notebook.

> **Prima di etichettare i commenti** leggere `docs/linee_guida_etichettatura.md`: le regole per
> decidere positivo, negativo o neutro (ogni commento da solo, senza contesto).

## Prima di cominciare

1. Aprire il terminale nella cartella principale del progetto (`SentimentProject`).
2. Attiva l'ambiente virtuale: `source venv/bin/activate` (a inizio riga compare `(venv)`).
3. (Se non ancora fatto) installare le librerie: `pip install -r requirements.txt`.
4. Compilare il file `.env` nella cartella principale, viene usato da `scraping.py`, con
   `INSTAGRAM_USER=...` e `INSTAGRAM_PASSWORD=...` (vedere `.env.example`).

Forma generale del comando:

```
python src/NOME_SCRIPT.py [argomenti] [--opzioni]
```

Per vedere le opzioni di uno script: `python src/NOME_SCRIPT.py --help` o `--h`.

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

**Esempio:**

```
python src/scraping.py uniurbit --posts 2 --comments 10
```

**Cosa produce** (due file in `data/raw/`):

- `posts_PROFILO.csv`: likes, comments, url, caption di ogni post
- `commenti_PROFILO.csv`: url_post, username, text, likes di ogni commento

Rilanciando lo script per lo stesso profilo i due file vengono **sovrascritti**: se si vuole
tenere la raccolta precedente, prima bisogna rinominarla.

**Da sapere**

- Chrome si apre da solo. Mentre legge i post meglio non muovere il mouse sopra la finestra.
- Al primo avvio fa il login e salva la sessione in `data/session/instagram_cookies.json`;
  dagli avvii successivi riusa i cookie, senza rifare il login. Quel file equivale alla
  password: non va mai caricato su GitHub.
- Se il login non funziona, prova con `--reset-session`. Troppi login ravvicinati
  fanno bloccare l'account da Instagram.
- Si può lanciare anche **dal notebook** (sezione 1, con `RUN_SCRAPING = True`): usa la stessa
  funzione, `scrape_profile`. Le credenziali restano nel `.env`, vengono lette dentro quella
  funzione e non compaiono mai né nel codice né negli output del notebook.

---

## `preprocessing.py`: prepara i dati e crea i set da etichettare

```
python src/preprocessing.py [--n N] [--seed S] [--set NN]
```

| Parametro | Obbligatorio | Default | Significato |
|---|---|---|---|
| `--n N` | no | 300 | quanti commenti estrarre per questo set |
| `--seed S` | no | 42 | seme del campionamento casuale (stesso seme e stessi dati = stesso campione) |
| `--set NN` | no | il primo libero | numero del set da creare (es. `03`) |

**Esempio: un set più piccolo**

```
python src/preprocessing.py --n 150
```

**Cosa fa**: legge tutti i `data/raw/commenti_*.csv`, li prepara, stampa quanti commenti sono
pronti (e quanti fatti solo di emoji) e crea un **nuovo set**,
`data/labeled/etichettati_a_mano_NN.csv` (NN = 00, 01, 02...), con tre colonne: `comment_id`,
`text_clean`, `label_true`. Compila `label_true` con `positivo`, `negativo` o `neutro`,
seguendo `docs/linee_guida_etichettatura.md`.

**Set numerati**

- I commenti di un set sono **diversi** da quelli degli altri set (non vengono estratti due
  volte).
- Un set esistente **non viene mai sovrascritto**. Se si chiede un numero già usato
  (`--set 00`) lo script lo dice e non fa nulla.
- Se restano meno commenti di `--n`, li prende tutti e avvisa. Se non ne restano, serve altro
  scraping.
- Gli altri script usano di default **tutti** i set; con `--sets 00 01` se ne scelgono solo alcuni.
- Se si ha già un file di etichette senza numero (`etichettati_a_mano.csv` o
  `da_etichettare.csv`), va rinominato in `etichettati_a_mano_00.csv`.
- Se si fa nuovamente lo scraping i commenti cambiano, ma le etichette restano agganciate ai commenti
  che ci sono ancora (il `comment_id` non cambia).

---

## `models.py`: classifica frasi o commenti con i modelli di sentiment

```
python src/models.py [MODELLI] [--text FRASE] [--dataset] [--sets NN ...] [--out FILE] [--neutral-threshold SOGLIA]
```

| Parametro | Obbligatorio | Default | Significato |
|---|---|---|---|
| `MODELLI` | no | tutti | quali modelli usare: nome intero o solo l'iniziale. `f` = feel_it, `b` = bert_italiano, `x` = xlmr_twitter, `n` = nlptown_stelle. Se se ne scrivono più di uno, vanno separati con uno spazio |
| `--text FRASE` | no | le 5 frasi fisse | frase da classificare (si può ripetere più volte) |
| `--dataset` | no | | esegue i modelli sui commenti etichettati, con entrambe le versioni del testo, e salva le predizioni in un CSV |
| `--sets NN [NN ...]` | no | tutti | quali set di etichette usare con `--dataset` (es. `--sets 00 01`) |
| `--out FILE` | no | `data/predictions/predizioni.csv` | dove salvare il CSV (solo con `--dataset`) |
| `--neutral-threshold SOGLIA` | no | | feel_it non ha la classe "neutro": con questa soglia (es. `0.7`) le sue predizioni poco sicure diventano "neutro" |

Ci sono due modalità.

**Prova rapida** (quella di default): classifica 5 frasi fisse, o le tue con `--text`, e
stampa il risultato di ogni modello. **Non crea nessun file.**

*Esempio 1: tutti i modelli sul dataset*

```
python src/models.py --dataset
```

*Esempio 2: tutti i modelli su una frase data*

```
python src/models.py --text "Bravo, complimenti!"
```

**Sul dataset** (`--dataset`): legge i commenti di `data/raw/`, li unisce alle etichette dei set
in `data/labeled/` (di default tutti, o quelli scelti con `--sets`) e classifica i commenti etichettati con ogni modello, sia su
`text_clean` sia su `text_emoji`. Salva un CSV con le colonne `comment_id`, `model`,
`version`, `label_pred`. Il CSV viene aggiornato dopo ogni modello, quindi un errore a metà non
fa perdere quanto già calcolato.

*Esempio 3: solo nlptown_stelle, salvando in un altro file*

```
python src/models.py n --dataset --out data/predictions/nlptown.csv
```

**Da sapere**

- Il primo avvio di ogni modello scarica i pesi da Hugging Face (centinaia di MB ciascuno,
  serve internet); dalle volte successive usa la copia salvata sul computer.
- Per `--dataset` servono prima lo scraping, il preprocessing e le etichette complete nei
  set: se manca qualcosa lo script si ferma e dice in quale file e a quali righe.
- Rilanciando `--dataset` con lo stesso `--out` il file viene **sovrascritto**: le predizioni
  vengono ricalcolate per tutti i commenti dei set scelti.
- Scrivere le iniziali dei modelli **prima** delle opzioni (`python src/models.py f x --dataset
  --sets 00`).

---

## `evaluate.py`: confronta le predizioni dei modelli con le tue etichette

```
python src/evaluate.py [--predictions FILE] [--sets NN ...] [--errors N] [--no-color] [--save] [--out-dir DIR]
```

| Parametro | Obbligatorio | Default | Significato |
|---|---|---|---|
| `--predictions FILE` | no | `data/predictions/predizioni.csv` | CSV creato da `models.py --dataset` |
| `--sets NN [NN ...]` | no | tutti | quali set di etichette usare (es. `--sets 00 01`) |
| `--errors N` | no | 0 | stampa **solo** N commenti sbagliati per ogni modello e versione del testo, **senza le statistiche** |
| `--no-color` | no | | niente colori (compaiono comunque solo se l'output va su un terminale) |
| `--save` | no | | salva tabelle (CSV) e grafici (PNG) in `--out-dir` |
| `--out-dir DIR` | no | `data/results` | dove salvare con `--save` |

**Esempio 1: tutte le tabelle a video**

```
python src/evaluate.py
```

**Esempio 2: solo gli errori, 5 per modello e versione**

```
python src/evaluate.py --errors 5
```

**Cosa stampa** (senza `--errors`)

1. Un riassunto: quanti commenti, quanti per classe, quanti con emoji, da quali set.
2. Le **metriche** per ogni modello e versione del testo (`text_clean` = emoji grezze,
   `text_emoji` = emoji tradotte): accuracy, macro-precision, macro-recall, macro-F1 e F1 di
   ogni classe. Le tabelle sono ripetute per sottoinsiemi: `tutti`, `con_emoji`, `solo_emoji` e
   `senza_neutro` (il confronto equo con feel_it, che non può dire "neutro"). Un
   sottoinsieme che coincide con `tutti` o è vuoto non compare.
3. **Emoji grezze o tradotte?** Un test di McNemar per ogni modello.
4. **Modelli a confronto**, due a due, per ogni versione del testo.

**Cosa stampa con `--errors N`**: solo gli errori, niente statistiche. Per ogni modello e versione
compaiono N commenti sbagliati scelti a caso, con l'etichetta, la predizione e il testo
esatto ricevuto dal modello; il titolo dice anche quanti errori ha fatto in totale.

**Colori** (solo nel terminale): ogni modello ha un colore diverso nel nome (il primo ciano, il
secondo magenta, il terzo giallo, il quarto blu); `positivo` è verde e `negativo` rosso. I
colori si spengono con `--no-color` o con la variabile `NO_COLOR`, e non vengono scritti se
l'output va su un file.

**Cosa salva `--save`** (in `data/results/`, sovrascrivendo i file precedenti): `metriche.csv`,
`mcnemar_versioni.csv`, `mcnemar_modelli.csv`, `confusioni.csv`, `predizioni_leggibili.csv`
(una riga per commento, con il testo, la tua etichetta e la risposta di ogni modello: si apre
con Excel o Numbers), `macro_f1.png`, `matrici_confusione.png`.

**Prima di lanciarlo** servono scraping, preprocessing, etichette complete e
`python src/models.py --dataset`.

---

## [!] Come leggere le metriche di `evaluate.py`

Il file delle predizioni contiene solo `comment_id` ed etichette: `evaluate.py` le rimette
insieme alle etichette e al testo, e ne ricava le misure descritte qui.

### Esempio che usiamo per tutte le spiegazioni

Un modello classifica 20 commenti. Righe = l'etichetta, colonne = la risposta del modello:

| | predetto positivo | predetto negativo | predetto neutro | totale |
|---|---|---|---|---|
| **vero positivo** | 8 | 1 | 1 | 10 |
| **vero negativo** | 1 | 4 | 1 | 6 |
| **vero neutro** | 2 | 0 | 2 | 4 |

Questa tabella si chiama **matrice di confusione** (`confusioni.csv`, grafici in
`matrici_confusione.png`). Sulla diagonale ci sono le risposte giuste; fuori diagonale si vede
**quali classi il modello scambia** tra loro.

### Accuracy

La quota di risposte giuste: **giuste / totali**. Nell'esempio: (8 + 4 + 2) / 20 = **0.70**.
È la misura più semplice, ma può ingannare (vedi più sotto).

### Precision, recall e F1 (si calcolano classe per classe)

Prendiamo la classe *positivo*:

- **Precision** = tra i commenti che il modello ha chiamato *positivo*, quanti lo erano davvero.
  Risponde a: "quando dice positivo, ha ragione?". Nell'esempio: 8 / (8 + 1 + 2) = 8 / 11 = **0.73**
  (il 2 e l'1 sono commenti negativi e neutri scambiati per positivi).
- **Recall** = tra i commenti *davvero positivi*, quanti il modello ne ha trovati.
  Risponde a: "trova tutti i positivi?". Nell'esempio: 8 / (8 + 1 + 1) = 8 / 10 = **0.80**.
- **F1** = unisce le due in un numero solo, con la media armonica: **2 · P · R / (P + R)**.
  Nell'esempio: 2 · 0.73 · 0.80 / (0.73 + 0.80) = **0.76**. La media armonica è bassa se
  *una* delle due è bassa: con precision 1.0 e recall 0.1 la media normale sarebbe 0.55, ma
  l'F1 è 0.18. Così un modello non può "compensare" una debolezza con un punto di forza.

Per le altre classi, nell'esempio:

| classe | precision | recall | F1 |
|---|---|---|---|
| positivo | 0.73 | 0.80 | 0.76 |
| negativo | 0.80 (4/5) | 0.67 (4/6) | 0.73 |
| neutro | 0.50 (2/4) | 0.50 (2/4) | 0.50 |

Nelle tabelle di `evaluate.py` le F1 delle singole classi sono `f1_positivo`, `f1_negativo` e
`f1_neutro`. Restano vuote se in quei commenti non c'è nessun esempio della classe.

### Macro (`macro_precision`, `macro_recall`, `macro_f1`)

La **media semplice** delle tre classi, con lo **stesso peso** a tutte, anche a quella con meno
commenti. Nell'esempio: macro-precision = (0.73 + 0.80 + 0.50) / 3 = **0.68**, macro-recall =
(0.80 + 0.67 + 0.50) / 3 = **0.66**, macro-F1 = (0.76 + 0.73 + 0.50) / 3 = **0.66**.
Contano solo le classi che esistono davvero nelle tue etichette: se in un sottoinsieme non ci
sono commenti neutri, la classe neutro non abbassa il risultato.

**Quale guardare per prima: la `macro_f1`**, perché non risente dello sbilanciamento tra classi.

### Perché non basta l'accuracy

Se il 90% dei commenti è positivo, un modello che risponde **sempre** "positivo" ha accuracy
**0.90** senza capire nulla: la sua macro-F1 è **0.47**, perché sulla classe negativa fa 0.
Con classi sbilanciate l'accuracy sopravvaluta i modelli; la macro-F1 no.

### Le colonne `n` e `subset`

- **n**: quanti commenti sono stati valutati. Con pochi commenti (meno di 30) le percentuali
  oscillano molto: lo script avvisa.
- **subset**: su quali commenti sono calcolate le metriche.
  - `tutti`: tutti i commenti etichettati.
  - `con_emoji`: solo quelli con almeno un'emoji. Sono gli unici in cui le due versioni del testo
    differiscono, quindi è qui che si vede l'effetto della traduzione delle emoji.
  - `solo_emoji`: solo quelli fatti esclusivamente di emoji.
  - `senza_neutro`: solo quelli che hai etichettato positivo o negativo. Serve per il confronto
    equo con feel_it, che sceglie solo tra positivo e negativo e non può mai dire "neutro".

### Test di McNemar: `solo_..._giusto` e `p_value`

Risponde a: "la differenza tra due serie di risposte è reale o è un caso?". Si usa per
confrontare due modelli, oppure lo stesso modello con le emoji grezze e tradotte, sugli
**stessi** commenti. Conta solo i commenti su cui i due **non** sono d'accordo:

- `solo_a_giusto` (o `solo_text_clean`): commenti giusti solo con A (o con le emoji grezze);
- `solo_b_giusto` (o `solo_text_emoji`): commenti giusti solo con B (o con le emoji tradotte).

I commenti su cui entrambi sono giusti, o entrambi sbagliati, non dicono nulla su chi sia
migliore e vengono ignorati. Se A e B fossero davvero equivalenti, i due numeri sarebbero
circa uguali. Il **p_value** è la probabilità di vedere per puro caso una divisione sbilanciata
come quella osservata (o di più). Convenzione: **p < 0.05** = differenza probabilmente reale;
sopra, può essere dovuta al caso.

*Esempio:* su 100 commenti A è giusto e B sbagliato in 10, B giusto e A sbagliato in 2: p = 0.039,
quindi A sembra davvero migliore. Un p alto **non** significa che i modelli sono uguali, ma che
con questi dati non si può dire.

### Bonferroni (`p_bonferroni`)

Con 4 modelli ci sono 6 confronti a coppie. Più confronti fai, più è probabile che uno risulti
"significativo" solo per caso. La correzione di Bonferroni **moltiplica ogni p_value per il numero
di confronti** (al massimo 1): è il valore da guardare nel confronto tra modelli.

*Esempio:* p = 0.039 su 6 confronti diventa 0.039 · 6 = **0.23**: la differenza non è più
dimostrata. Il `p_value` grezzo va letto solo per il singolo confronto "emoji grezze o
tradotte?" di un modello.

### Altre colonne

- **`acc_text_clean`, `acc_text_emoji`** (in `mcnemar_versioni.csv`): accuracy dello stesso modello
  con le emoji grezze e con quelle tradotte.
- **`giuste`, `totali`** (in `predizioni_leggibili.csv`): per ogni commento, quante predizioni
  su quante (modelli per versioni del testo) sono corrette. Ordinando per `giuste` si vedono i
  commenti più difficili: se **tutte** sono sbagliate, il commento è ambiguo oppure la tua
  etichetta va ricontrollata.
- **`set`**: il set di etichette da cui viene il commento.
- **Kappa di Cohen** (solo nel notebook, sezione 3): non misura i modelli, ma l'accordo tra **te
  e un'altra persona** nell'etichettare gli stessi commenti, tolto l'accordo dovuto al caso.
  Orientativamente: 0.41-0.60 moderato, 0.61-0.80 sostanziale, oltre 0.80 quasi perfetto.

---

## Problemi comuni

| Messaggio | Cosa fare |
|---|---|
| `ModuleNotFoundError` | l'ambiente virtuale non è attivo: `source venv/bin/activate` |
| `Credenziali mancanti: controlla il file .env` | controllare che `.env` esista nella cartella principale e che abbia i due valori |
| `Nessun file 'commenti_*.csv'` | lanciare prima `scraping.py`: serve almeno un file in `data/raw/` |
| `Interrotto: login non riuscito` | provare `--reset-session`; controllare credenziali o richieste di verifica sul telefono |
| `Modello 'z' sconosciuto` | usare il nome intero o l'iniziale giusta: `f`, `b`, `x`, `n` |
| `Predizioni non trovate` | creare prima le predizioni: `python src/models.py --dataset` |
| `Impossibile partire: Etichette da correggere` | completare o correggere le righe indicate nel file indicato (valori validi: `positivo`, `negativo`, `neutro`) |
| `Nessun set di etichette` | creare un set con `python src/preprocessing.py`; se si ha un vecchio file di etichette, rinominarlo `etichettati_a_mano_00.csv` |
| `Il set NN non esiste` | il messaggio elenca i set disponibili (es. `00, 01`) |
| `Tutti i commenti sono già in un set di etichette` | non restano commenti da estrarre: serve altro scraping |
| `commenti compaiono più volte nei set di etichette` | uno stesso commento è in due set (o due volte nello stesso): togliere il doppione dal file |