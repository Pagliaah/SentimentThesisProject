# =====================================================================
# scraping.py - raccolta dei commenti Instagram per la tesi
#
# Flusso (nello stesso ordine in cui sono scritti i metodi qui sotto):
#   1. si apre Chrome
#   2. login (cookie salvati, o credenziali del .env)
#   3. lista dei post di un profilo (url, like, n. commenti, caption)
#   4. commenti di ogni post
#   5. salvataggio in data/raw/*.csv
#
# Uso (dalla cartella principale del progetto, con il venv attivo):
#   python src/scraping.py NOME_PROFILO --posts 10 --comments 50
# Dal notebook: from src import scraping; scraping.scrape_profile("NOME_PROFILO", 10, 50)
# =====================================================================

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from selenium import webdriver
from selenium.common.exceptions import (NoSuchElementException, StaleElementReferenceException,
                                        TimeoutException, WebDriverException)
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


# ---------------------------------------------------------------------
# PERCORSHI
# Calcolati a partire da dove si trova questo file, cosi' funzionano
# anche se lanci lo script da un'altra cartella.
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
# Cookie della sessione Instagram. ATTENZIONE: chi li ha entra nell'account senza password,
# quindi questo file non deve MAI finire su GitHub (la cartella data/ e' ignorata da git).
SESSION_FILE = ROOT_DIR / "data" / "session" / "instagram_cookies.json"


# ---------------------------------------------------------------------
# SELETTORI
# Instagram genera le classi CSS in automatico (x78zum5, _aagu, ...) e
# le cambia spesso. Se un giorno lo scraper smette di trovare post o
# commenti, il problema e' quasi sempre qui: aggiorna SOLO queste righe
# (leggendo i nuovi nomi con F12 -> "Ispeziona" nel browser).
# ---------------------------------------------------------------------
COOKIE_XPATH = ("//button[text()='Consenti tutti i cookie' or text()='Consenti cookie facoltativi' "
                "or text()='Accetta tutto' or text()='Allow all cookies']")
LOGIN_BUTTON_XPATH = "//button[text()='Accedi' or text()='accedi']"
# Pagina che richiede il login: se la sessione non e' valida Instagram reindirizza a /accounts/login/
SESSION_CHECK_URL = "https://www.instagram.com/accounts/edit/"
LOGIN_URL_PART = "accounts/login"
USERNAME_FIELD_NAME = "email"      # attributo name del campo username nella pagina di login
PASSWORD_FIELD_NAME = "pass"       # attributo name del campo password
LOGIN_OK_XPATH = ("//div[contains(text(), 'Salva le informazioni')] "
                  "| //button[text()='Non ora' or text()='Not Now'] "
                  "| //div[@role='button' and (text()='Non ora' or text()='Not Now')]")

POST_THUMB_CLASS = "_aagu"                           # miniatura di un post nella griglia del profilo
POST_STATS_XPATH = "./following-sibling::div//ul/li"  # like e commenti che compaiono al passaggio del mouse

COMMENT_BLOCK_CSS = ".x78zum5.xdt5ytf.x1iyjqo2"                      # contenitore di un commento
COMMENT_REPLY_XPATH = ".//*[text()='Rispondi' or text()='Reply']"    # tasto "Rispondi" (solo nei veri commenti)
COMMENT_FIELDS_CSS = ".x1lliihq.x1plvlek.xryxfnj.x1n2onr6.xyejjpt"   # [0]=username, [1]=data, [2]=testo
COMMENT_LIKES_CSS = ".x1lliihq.x193iq5w.x6ikm8r.x10wlt62.xlyipyv.xuxw1ft"  # numero di like del commento
PAGE_REPLY_XPATH = "//*[text()='Rispondi' or text()='Reply']"    # come sopra, ma cerca in tutta la pagina

# Un username Instagram puo' contenere solo lettere, numeri, punti e underscore.
# Serve a scartare i falsi "commenti" (es. "profilo_1\ne\nprofilo_2" nei post in collaborazione).
USERNAME_RE = re.compile(r"[A-Za-z0-9._]+")

# ---------------------------------------------------------------------
# SCROLL DEI COMMENTI
# Instagram carica i commenti a blocchi (circa 10 alla volta) mentre scorri
# la sezione commenti: senza scroll se ne leggono solo i primi.
# ---------------------------------------------------------------------
MAX_SCROLL_ROUNDS = 60       # limite di sicurezza: massimo di scroll per ogni post
STALL_ROUNDS = 3             # ci si ferma dopo tanti scroll di fila senza nuovi commenti
SCROLL_PAUSE = (1.5, 2.5)    # attesa (secondi) dopo ogni scroll, per dare tempo di caricare

# JavaScript eseguito nel browser. Riceve un elemento che sta dentro ai commenti
# (arguments[0]) e risale i suoi genitori fino al primo che e' "scorrevole"
# (overflow auto/scroll e contenuto piu' alto del box): quello e' il pannello dei
# commenti, e lo porta in fondo. Non usa nomi di classe, quindi non si rompe quando
# Instagram li cambia. Se non trova nessun pannello (finestra stretta: scorre
# tutta la pagina) scorre la pagina intera.
SCROLL_JS = """
var node = arguments[0].parentElement;
while (node) {
    var overflowY = window.getComputedStyle(node).overflowY;
    if ((overflowY === 'auto' || overflowY === 'scroll') && node.scrollHeight > node.clientHeight) {
        node.scrollTop = node.scrollHeight;
        return node.scrollHeight;
    }
    node = node.parentElement;
}
window.scrollTo(0, document.body.scrollHeight);
return document.body.scrollHeight;
"""


# Legge username e password dal file .env (nella cartella principale)
# e li restituisce. Se mancano, ferma subito il programma con un messaggio
# chiaro, invece di fallire piu' avanti durante il login.
# Le credenziali NON vanno mai scritte nel codice o nei notebook.
def get_credentials():
    load_dotenv(ROOT_DIR / ".env")
    user = os.getenv("INSTAGRAM_USER")
    password = os.getenv("INSTAGRAM_PASSWORD")
    if not user or not password:
        raise RuntimeError("Credenziali mancanti: controlla il file .env "
                           "(INSTAGRAM_USER e INSTAGRAM_PASSWORD)")
    return user, password


class InstagramScraper:

    # Costruttore: apre una finestra di Chrome controllata da Selenium.
    # ChromeDriverManager scarica da solo il chromedriver compatibile con il
    # tuo Chrome. I cookie vengono cancellati per partire da una sessione pulita.
    # La finestra puo' anche non essere in primo piano, ma durante lo scraping
    # dei post NON muovere il mouse sopra Chrome: il "hover" usato per leggere
    # like e commenti verrebbe disturbato.
    def __init__(self):
        self.browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        self.link = "https://www.instagram.com"
        self.browser.delete_all_cookies()

    # Chiude Chrome. Da chiamare sempre alla fine (vedi try/finally in main).
    def close(self):
        self.browser.quit()

    # Pausa casuale tra due valori (in secondi). Le pause di durata variabile
    # simulano un utente umano e riducono il rischio che Instagram limiti
    # l'account (shadowban).
    def _pausa(self, minimo=2, massimo=3):
        time.sleep(random.uniform(minimo, massimo))

    # ----- Metodi di attesa -----
    # Instagram carica la pagina a pezzi, quindi prima di leggere un elemento
    # bisogna aspettare (max 10 secondi) che esista. Se non compare, Selenium
    # solleva TimeoutException. `by` indica come cercare (By.XPATH, By.NAME,
    # By.CSS_SELECTOR, ...), `value` e' il selettore.

    # Aspetta che esista UN elemento e lo restituisce.
    def wait_for_object(self, by, value):
        return WebDriverWait(self.browser, 10).until(ec.presence_of_element_located((by, value)))

    # Aspetta che esistano degli elementi e li restituisce TUTTI in una lista.
    def wait_for_objects(self, by, value):
        return WebDriverWait(self.browser, 10).until(ec.presence_of_all_elements_located((by, value)))

    # Aspetta che un elemento sia presente E cliccabile (per i bottoni).
    def wait_for_click(self, by, value):
        return WebDriverWait(self.browser, 10).until(ec.element_to_be_clickable((by, value)))

    # ----- Sessione (cookie salvati) -----

    # Salva su disco i cookie della sessione appena aperta (SESSION_FILE), cosi' alla
    # prossima esecuzione si puo' rientrare senza rifare il login: troppi login in
    # poco tempo fanno bloccare l'account da Instagram. file non versionato, leggibile 
    # solo in locale.
    def _save_session(self):
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        SESSION_FILE.write_text(json.dumps(self.browser.get_cookies()), encoding="utf-8")
        try:
            os.chmod(SESSION_FILE, 0o600)
        except OSError:
            pass    # ad esempio su Windows: i permessi funzionano diversamente
        print(f"Sessione salvata in {SESSION_FILE.relative_to(ROOT_DIR)}")

    # Prova a rientrare in Instagram con i cookie salvati, senza login. Passi:
    #   1. se il file non c'e' (o e' illeggibile) restituisce subito False;
    #   2. apre Instagram (i cookie si possono aggiungere solo a un sito gia' aperto)
    #      e li inserisce uno a uno, saltando quelli che il browser rifiuta;
    #   3. verifica che la sessione sia ancora valida aprendo una pagina che richiede
    #      il login (SESSION_CHECK_URL): se Instagram reindirizza a /accounts/login/,
    #      oppure compare il campo username, la sessione e' scaduta.
    # Restituisce True se la sessione e' valida. Altrimenti cancella i cookie caricati
    # (per non disturbare il login normale) e restituisce False.
    # Nota: quando la sessione e' valida il controllo attende fino a 6 secondi il
    # reindirizzamento che non arriva: e' normale, succede una volta per esecuzione.
    def _login_with_saved_session(self):
        if not SESSION_FILE.exists():
            return False
        try:
            cookies = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print("File della sessione illeggibile: faccio un login normale.")
            return False

        self.browser.get(self.link)
        added = 0
        for cookie in cookies:
            try:
                self.browser.add_cookie(cookie)
                added += 1
            except WebDriverException:
                pass
        if added == 0:
            return False

        self.browser.get(SESSION_CHECK_URL)
        try:
            WebDriverWait(self.browser, 6).until(lambda d: LOGIN_URL_PART in d.current_url)
            expired = True
        except TimeoutException:
            expired = bool(self.browser.find_elements(By.NAME, USERNAME_FIELD_NAME))

        if expired:
            print("Sessione salvata scaduta: faccio un login normale.")
            self.browser.delete_all_cookies()
            return False

        print("Sessione ripresa dai cookie salvati: nessun login necessario.")
        return True

    # ----- Login -----

    # Entra in Instagram. Prima prova con i cookie salvati (_login_with_saved_session);
    # se non ci sono o sono scaduti, esegue il login con username e password. Passi:
    #   1. apre la home e accetta i cookie (se compare il popup);
    #   2. clicca "Accedi" se la pagina di login non si e' aperta da sola;
    #   3. scrive username e password nei campi e preme Invio;
    #   4. attende fino a 20 secondi il popup "Salva le informazioni / Non ora",
    #      che compare solo se il login e' riuscito; a quel punto salva la sessione.
    # Restituisce True se si e' dentro, False altrimenti (password errata, richiesta
    # di verifica di sicurezza, connessione lenta...).
    def login(self, username, password):
        if self._login_with_saved_session():
            return True

        self.browser.get(self.link)

        try:
            self.wait_for_click(By.XPATH, COOKIE_XPATH).click()
            print("Cookie accettati.")
        except TimeoutException:
            print("Nessun popup cookie trovato, procedo.")

        self._pausa()

        # Non sempre serve: se la pagina di login e' gia' aperta il bottone
        # non esiste e si passa oltre.
        try:
            self.wait_for_click(By.XPATH, LOGIN_BUTTON_XPATH).click()
        except TimeoutException:
            print("Bottone 'Accedi' non trovato (login gia' aperto?), procedo.")

        username_input = self.wait_for_objects(By.NAME, USERNAME_FIELD_NAME)[0]
        password_input = self.wait_for_objects(By.NAME, PASSWORD_FIELD_NAME)[0]
        username_input.send_keys(username)
        password_input.send_keys(password)
        password_input.send_keys(Keys.ENTER)

        try:
            WebDriverWait(self.browser, 20).until(
                ec.presence_of_element_located((By.XPATH, LOGIN_OK_XPATH)))
            print("Login effettuato.")
            self._save_session()
            return True
        except TimeoutException:
            print("Il popup 'Salva info' non e' comparso: login probabilmente fallito "
                  "(controlla credenziali o eventuale verifica di sicurezza).")
            return False

    # ----- Utilita' -----

    # Trasforma i numeri letti da Instagram (testo) in interi.
    # Esempi: "30.400" -> 30400 | "12,4 Mln" -> 12400000 | "n/d" -> 0.
    # Il punto (separatore delle migliaia in italiano) e la virgola vengono
    # rimossi: per questo "12,4 Mln" diventa "124 mln", e moltiplicando per
    # 100000 (uno zero in meno, voluto) si ottiene proprio 12.400.000.
    # "mln" va controllato PRIMA di "m", altrimenti verrebbe tagliato male.
    # I casi "k" e "m" non dovrebbero servire (Instagram scrive i numeri
    # per esteso) e con decimali darebbero risultati sbagliati.
    # Se il testo non e' interpretabile restituisce 0.
    def format_numbers(self, number):
        number = str(number).lower().strip()
        if number == "n/d" or not number:
            return 0
        try:
            number = number.replace(",", "").replace(".", "")
            if "mln" in number:
                return int(float(number.replace("mln", "")) * 100000)
            if "k" in number:
                return int(float(number.replace("k", "")) * 1000)
            if "m" in number:
                return int(float(number.replace("m", "")) * 1000000)
            return int(float(number))
        except ValueError:
            return 0

    # ----- Lista dei post -----

    # Raccoglie i post di un profilo pubblico e restituisce un DataFrame con
    # le colonne: likes, comments, url, caption.
    # `top_n` e' quanti post vuoi alla fine (oppure "all" per tutti).
    # Per ottenere i post con piu' commenti ne vengono letti un po' di piu'
    # (con top_n=20 se ne leggono 33; con 100, 120) e poi, se
    # `sort_by_comments=True`, si tengono i top_n con piu' commenti. Il margine
    # serve anche perche' qualche elemento della griglia puo' non essere un post.
    # Come funziona il ciclo: Instagram carica i post a blocchi mentre scorri.
    # Si leggono le miniature visibili e, passando il mouse su ognuna
    # (_scrape_single_post), la pagina scorre da sola e ne compaiono di nuove.
    # Ad ogni giro si confrontano le miniature gia' lette con quelle presenti
    # ora: se non ce ne sono di nuove, il profilo e' finito.
    def get_metadata(self, profile, top_n=10, sort_by_comments=True):
        self.browser.get(f"{self.link}/{profile}")

        if top_n == "all":
            limit = float("inf")
            print("--- Estrazione di tutti i post in corso...")
        else:
            limit = int(top_n + math.log10(top_n) * 10)
            print(f"--- Obiettivo: {top_n} post. Ne leggo {limit}")

        rows = []
        processed = []      # miniature gia' lette
        print("Progresso:", end=" ")

        while len(rows) < limit:
            try:
                pictures = self.wait_for_objects(By.CLASS_NAME, POST_THUMB_CLASS)
            except TimeoutException:
                print(f"\nNessun post trovato per '{profile}' (profilo privato o inesistente?)")
                break

            new_pictures = [p for p in pictures if p not in processed]
            if not new_pictures:        # nessuna miniatura nuova: fine del profilo
                break

            for picture in new_pictures:
                if len(rows) >= limit:
                    break
                rows.append(self._scrape_single_post(picture))
                processed.append(picture)
                print("|", end=(" " if len(rows) % 5 == 0 else ""))
        print()

        df = pd.DataFrame(rows, columns=["likes", "comments", "url", "caption"])
        df["likes"] = df["likes"].map(self.format_numbers)
        df["comments"] = df["comments"].map(self.format_numbers)

        if sort_by_comments:
            df = df.sort_values("comments", ascending=False)
        if top_n != "all":
            df = df.head(top_n)
        return df.reset_index(drop=True)

    # Legge i dati di UNA miniatura e li restituisce in un dizionario.
    # 1. Hover: passando il mouse sulla miniatura Instagram mostra un
    #    overlay con like e commenti (e fa scorrere la pagina fino all'elemento).
    # 2. Like e commenti: l'overlay e' una lista (<ul><li>); il primo <li> contiene
    #    i like, il secondo i commenti. L'XPath "./following-sibling::div//ul/li"
    #    significa: dalla miniatura, vai al <div> fratello, scendi fino alla lista.
    # 3. Url e caption: l'url e' nell'attributo href del link <a> che contiene la
    #    miniatura ("./.." = elemento genitore); la caption e' nell'attributo
    #    alt dell'immagine.
    # Se qualcosa non si trova non si ferma tutto: mette valori di ripiego.
    def _scrape_single_post(self, picture):
        ActionChains(self.browser).move_to_element(picture).perform()

        likes_raw, comments_raw = "0", "0"
        try:
            stats = picture.find_elements(By.XPATH, POST_STATS_XPATH)
            if len(stats) >= 2:
                likes_raw = stats[0].find_element(By.TAG_NAME, "span").text
                comments_raw = stats[1].find_element(By.TAG_NAME, "span").text
        except Exception as e:
            print(f"\nErrore lettura like/commenti: {e}")

        try:
            url = picture.find_element(By.XPATH, "./..").get_attribute("href")
            caption = picture.find_element(By.TAG_NAME, "img").get_attribute("alt")
        except Exception:
            url, caption = "N/D", ""

        return {"likes": likes_raw, "comments": comments_raw, "url": url, "caption": caption}

    # ----- Commenti -----

    # Raccoglie i commenti di una lista di post e restituisce un DataFrame con
    # le colonne: url_post, username, text, likes.
    # `post_urls`: gli url dei post (colonna "url" di get_metadata).
    # `max_per_post`: massimo di commenti PER OGNI post (oppure "all").
    # `post_owner`: username del proprietario del profilo; serve a scartare la
    # didascalia del post, che nella pagina ha gli stessi tag di un commento.
    # Attenzione: legge solo i commenti di primo livello (non le risposte).
    def get_comments(self, post_urls, max_per_post, post_owner):
        limit = float("inf") if max_per_post == "all" else max_per_post
        print(f"--- Estrazione commenti da {len(post_urls)} post (max {max_per_post} per post)...")

        rows = []
        for i, url in enumerate(post_urls, start=1):
            rows.extend(self._scrape_single_post_comment(url, limit, post_owner))
            print(f"--- Post {i}/{len(post_urls)} completato. Commenti totali finora: {len(rows)}")

        return pd.DataFrame(rows, columns=["url_post", "username", "text", "likes"])

    # Carica i commenti di un post scorrendo il pannello dei commenti.
    # Ad ogni giro:
    #   1. conta i commenti caricati (= quanti tasti "Rispondi" ci sono nella pagina);
    #   2. si ferma se ne ha abbastanza (limit + 10 di margine, perche' alcuni
    #      verranno scartati dai filtri) oppure se non e' comparso nulla di nuovo
    #      per STALL_ROUNDS giri di fila (commenti finiti, o connessione lenta);
    #   3. altrimenti scorre in fondo (SCROLL_JS) partendo dall'ultimo commento
    #      e aspetta che Instagram carichi il blocco successivo.
    # MAX_SCROLL_ROUNDS impedisce che il ciclo giri all'infinito.
    # Restituisce il numero di commenti caricati.
    def _scroll_comments(self, limit):
        target = limit + 10
        loaded = 0
        previous = -1
        stalled = 0

        for _ in range(MAX_SCROLL_ROUNDS):
            replies = self.browser.find_elements(By.XPATH, PAGE_REPLY_XPATH)
            loaded = len(replies)

            if loaded == 0 or loaded >= target:
                break

            stalled = stalled + 1 if loaded == previous else 0
            if stalled >= STALL_ROUNDS:
                break
            previous = loaded

            try:
                self.browser.execute_script(SCROLL_JS, replies[-1])
            except StaleElementReferenceException:
                pass    # la pagina si e' aggiornata proprio ora: si riprova al giro dopo
            self._pausa(*SCROLL_PAUSE)

        print(f"Commenti caricati nella pagina: {loaded}")
        return loaded

    # Apre UN post, carica altri commenti con lo scroll (_scroll_comments) e li estrae,
    # restituendo una lista di dizionari {url_post, username, text, likes}.
    # Per ogni blocco "commento" della pagina:
    #   1. se non contiene il tasto "Rispondi" non e' un commento: si salta;
    #   2. si leggono username (campo 0) e testo (campo 2; il campo 1 e' la data,
    #      non usata);
    #   3. si scartano i casi anomali:
    #        - testo vuoto o uguale all'username (errore di lettura);
    #        - commento gia' visto nello stesso post (stesso utente e stesso testo);
    #        - username non valido (a capo, spazi): post in collaborazione ("profilo_1 e profilo_2");
    #        - la didascalia del proprietario del post;
    #   4. si legge il numero di like del commento con una regex (prende il
    #      primo numero del testo, 0 se non c'e');
    #   5. ci si ferma quando si raggiunge `limit` commenti.
    # Se un singolo commento da' errore si passa al successivo, senza perdere
    # quelli gia' raccolti.
    def _scrape_single_post_comment(self, url, limit, post_owner):
        self.browser.get(url)
        self._pausa()    # un utente vero guarda il post prima di scorrere
        self._scroll_comments(limit)    # carica altri commenti oltre ai primi

        results = []
        seen = set()     # coppie (username, testo) gia' viste in questo post

        try:
            comment_blocks = self.browser.find_elements(By.CSS_SELECTOR, COMMENT_BLOCK_CSS)
            print(f"Trovati {len(comment_blocks)} blocchi HTML nel post.")

            for block in comment_blocks:
                if len(results) >= limit:
                    break
                try:
                    try:
                        block.find_element(By.XPATH, COMMENT_REPLY_XPATH)
                    except NoSuchElementException:
                        continue

                    fields = block.find_elements(By.CSS_SELECTOR, COMMENT_FIELDS_CSS)
                    user = fields[0].text.strip()
                    text = fields[2].text.strip()

                    if not text or text == user:
                        continue
                    if (user, text) in seen:
                        continue
                    if not USERNAME_RE.fullmatch(user):    # collaborazioni ("a\ne\nb") e campi sbagliati
                        continue
                    if post_owner and post_owner in user and post_owner in text:
                        continue

                    likes = 0
                    try:
                        likes_text = block.find_element(By.CSS_SELECTOR, COMMENT_LIKES_CSS).text.strip()
                        found = re.findall(r"\d+[.,\d]*", likes_text)
                        if found:
                            likes = int(float(found[0].replace(".", "").replace(",", ".")))
                    except Exception:
                        likes = 0

                    seen.add((user, text))
                    results.append({"url_post": url, "username": user, "text": text, "likes": likes})

                except Exception:
                    continue

            print(f"Commenti validi estratti da questo post: {len(results)}")

        except Exception as e:
            print(f"Errore scraping commenti di {url}: {e}")

        return results


# Raccoglie post e commenti di un profilo e li salva in data/raw/ (posts_PROFILO.csv e
# commenti_PROFILO.csv, sovrascrivendo quelli dello stesso profilo). Passi: legge le
# credenziali dal file .env, apre Chrome, fa login (o riprende la sessione salvata),
# legge i post, legge i commenti e chiude Chrome (anche se qualcosa va storto).
# Ha la stessa funzione dalla riga di comando (main) e dal notebook.
#
# SICUREZZA: username e password vengono letti QUI DENTRO dal file .env e restano variabili
# locali di questa funzione: non vengono mai stampati, restituiti o salvati, quindi chi la
# chiama (anche un notebook) non li vede mai. Restituisce i due DataFrame (post, commenti):
# i commenti contengono gli username VERI, quindi non mostrarli nel notebook.
# Solleva RuntimeError (con un messaggio chiaro, senza dati segreti) se mancano le credenziali
# o se il login non riesce.
def scrape_profile(profile, n_posts=10, n_comments=50, reset_session=False):
    if reset_session and SESSION_FILE.exists():
        SESSION_FILE.unlink()
        print("Sessione salvata cancellata.")

    user, password = get_credentials()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    bot = InstagramScraper()
    try:
        if not bot.login(user, password):
            raise RuntimeError("Interrotto: login non riuscito (controlla le credenziali nel "
                               ".env o eventuali richieste di verifica sul telefono).")

        posts = bot.get_metadata(profile, n_posts)
        posts.to_csv(RAW_DIR / f"posts_{profile}.csv", index=False)
        print(f"Salvati {len(posts)} post in data/raw/posts_{profile}.csv")

        urls = [u for u in posts["url"] if u and u != "N/D"]
        comments = bot.get_comments(urls, n_comments, profile)
        comments.to_csv(RAW_DIR / f"commenti_{profile}.csv", index=False)
        print(f"Salvati {len(comments)} commenti in data/raw/commenti_{profile}.csv")
    finally:
        bot.close()

    return posts, comments


# Punto di ingresso: viene eseguito solo quando lanci direttamente
# `python src/scraping.py ...`, non quando il file viene importato.
# Legge i parametri dalla riga di comando e chiama scrape_profile.
def main():
    parser = argparse.ArgumentParser(description="Scraping dei commenti di un profilo Instagram")
    parser.add_argument("profile", help="username del profilo (senza @)")
    parser.add_argument("--posts", type=int, default=10, help="quanti post leggere (default 10)")
    parser.add_argument("--comments", type=int, default=50, help="max commenti per post (default 50)")
    parser.add_argument("--reset-session", action="store_true",
                        help="cancella i cookie salvati e rifa' il login con username e password")
    args = parser.parse_args()

    try:
        scrape_profile(args.profile, args.posts, args.comments, args.reset_session)
    except RuntimeError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()