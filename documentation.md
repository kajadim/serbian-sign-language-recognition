# Prepoznavanje srpske azbuke znakovnog jezika

Projekat za prepoznavanje slova srpske azbuke izgovorenih znakovnim jezikom (SSL — Serbian Sign Language) korišćenjem snimaka pokreta ruku i tela sa kamere. Sistem koristi **MediaPipe Holistic** za izdvajanje koordinata zglobova ruku i tela iz video snimka, a zatim **LSTM neuronsku mrežu** koja na osnovu niza od 40 frejmova prepoznaje koje slovo je pokazano. Projekat uključuje i alat za snimanje sopstvenih primera i alat za prepoznavanje u realnom vremenu preko kamere.

Skup podataka je preuzet iz repozitorijuma [mlradak/serbian-sign-language](https://github.com/mlradak/serbian-sign-language), gde su slova srpske azbuke organizovana u foldere, a svaki snimak jednog znaka je sačuvan kao `.npy` fajl koji sadrži niz frejmova sa koordinatama tela i šaka (33 tačke tela × 4 vrednosti + 21 tačka leve šake × 3 vrednosti + 21 tačka desne šake × 3 vrednosti = 258 vrednosti po frejmu).

## Kako funkcioniše projekat

Projekat je podeljen u nekoliko faza koje prirodno idu jedna za drugom: **snimanje/učitavanje podataka → obrada i ekstrakcija feature-a → treniranje modela → podešavanje hiperparametara → evaluacija → prepoznavanje u realnom vremenu**. Svaka faza je realizovana kroz jednu ili više Python skripti.

### 1. Prikupljanje podataka — `record_samples.py`

Skripta otvara kameru i pomoću **MediaPipe Holistic** modela u realnom vremenu prati pozu tela i obe šake. Radi po principu automata sa stanjima (_state machine_):

- **`waiting`** — sistem čeka da se ruka pojavi u kadru.
- **`recording`** — čim se ruka detektuje, počinje snimanje. Snima se tačno **40 frejmova**; svaki frejm se pretvara u niz od 258 brojeva (koordinate tela i obe šake) funkcijom `extract_keypoints()`.
- Ako ruka izađe iz kadra previše rano (pre nego što se skupi 40 frejmova), snimak se odbacuje kao neuspešan.
- Kada se skupi 40 frejmova, snimak se čuva kao `.npy` fajl u folder `data/<SLOVO>/`, sa imenom fajla po trenutnom vremenskom žigu.

Pokreće se sa slovom kao argumentom, npr.:

```bash
python record_samples.py D
```

i snima nove primere za slovo **D**. Na ekranu se prikazuju i skeleton tela/šaka, trenutno stanje, broj snimljenih primera i progres trenutnog snimka. Snimanje se prekida tasterom **ESC**.

### 2. Učitavanje i analiza podataka — `load_and_explore_data.py`

Ova skripta učitava sirove `.npy` fajlove iz foldera `data/` (svaki podfolder predstavlja jedno slovo/klasu) i vrši osnovnu eksploraciju skupa podataka:

- `load_npy_file()` — učitava jedan `.npy` fajl i deli ravan niz brojeva na frejmove fiksne veličine (258 vrednosti po frejmu).
- `load_dataset()` — prolazi kroz sve foldere (slova) i učitava sve snimke u liste `X` (frejmovi) i `y` (labela/slovo).
- `analyze_dataset()` — ispisuje broj snimaka po slovu, statistiku broja frejmova po snimku (min/max/prosek) i generiše grafikon (`dataset_analysis.png`) sa raspodelom broja snimaka po slovu i histogramom dužine snimaka.

Ova skripta se uglavnom koristi da se provеri da li je skup podataka balansiran i ispravno učitan, pre nego što se krene na obradu i treniranje.

### 3. Obrada podataka — pomoćni moduli

Pre nego što se podaci koriste za treniranje, prolaze kroz tri koraka obrade. Svaki korak je u svojoj skripti, jer se isti kod koristi i kasnije u `realtime.py` za prepoznavanje uživo.

**`interpolation.py`** — Kamera ponekad ne uspe da detektuje šaku u nekim frejmovima (npr. kad je ruka brzo prošla ili je delimično van kadra), pa taj frejm ostane popunjen nulama. Funkcija `interpolate_sequence_gaps()` pronalazi takve "rupe" u nizu frejmova i popunjava ih linearnom interpolacijom između poslednjeg poznatog i prvog sledećeg poznatog položaja šake. Ako su frejmovi na samom početku ili kraju snimka nedostupni (nema prethodne/sledeće poznate vrednosti), ti frejmovi se ne popunjavaju.

**`body_normalization.py`** — Različiti ljudi snimaju znakove na različitoj udaljenosti od kamere i u različitim pozicijama u kadru, što bi modelu otežalo učenje. Funkcija `normalize_sequence()` rešava to tako što za svaki frejm:

- izračuna centar (sredinu) i razmeru (_scale_) na osnovu položaja **levog i desnog ramena**,
- pomeri sve koordinate (tela i obe šake) tako da centar bude u koordinatnom početku,
- skalira sve koordinate prema razmeri ramena, čime se uklanja uticaj udaljenosti od kamere.

**`curl_smoothing.py`** — Sadrži samo jednu pomoćnu funkciju, `smooth_curl_sequence()`, koja primenjuje _rolling average_ (klizni prosek) preko prozora od nekoliko frejmova, kako bi se uklonio šum (treperenje) u izračunatim feature-ima savijenosti prstiju kroz vreme.

### 4. Priprema finalnog skupa podataka — `prepare_dataset.py`

Ovo je centralna skripta koja spaja sve korake obrade i pravi finalni skup podataka spreman za treniranje:

1. Učitava sirove podatke pomoću `load_dataset()` iz `load_and_explore_data.py`.
2. Za svaki snimak (`process_sequence_v3()`):
   - popunjava nedostajuće frejmove (`interpolate_sequence_gaps`),
   - normalizuje koordinate u odnosu na ramena (`normalize_sequence`),
   - računa **10 dodatnih feature-a savijenosti prstiju** (_finger curl_) — za svaki od 5 prstiju na obe šake računa odnos rastojanja vrh-zglob i osnova-zglob (`finger_curl()`), što opisuje koliko je prst savijen, i taj niz feature-a po vremenu izglađuje (`smooth_curl_sequence`).
   - Konačno, svaki frejm ima **268 vrednosti** (258 originalnih koordinata + 10 curl feature-a).
3. Sve sekvence se skaliraju na opseg [0, 1] pomoću `MinMaxScaler`-a iz scikit-learn-a.
4. Labele (slova) se kodiraju u brojeve pomoću `LabelEncoder`-a.
5. Skup podataka se deli na **trening (≈70%), validacioni (≈15%) i test (≈15%)** skup, uz `stratify` da bi svako slovo bilo ravnomerno zastupljeno u sva tri dela.
6. Svi nizovi (`X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`), lista klasa (`classes.npy`) i sam scaler (`scaler.pkl`) se čuvaju u folder `models/`, da bi mogli da se ponovo koriste u kasnijim skriptama bez ponovne obrade.

### 5. Treniranje modela — `train_model.py`

Učitava pripremljene `.npy` fajlove iz `models/` i trenira **LSTM neuronsku mrežu** (Keras/TensorFlow) sledeće arhitekture:

```
LSTM(128, return_sequences=True) → Dropout(0.3)
→ LSTM(64) → Dropout(0.3)
→ Dense(64, activation='relu')
→ Dense(broj_klasa, activation='softmax')
```

Model se kompajlira sa `Adam` optimizatorom (learning rate 0.0005) i `categorical_crossentropy` funkcijom greške. Koristi se:

- **`EarlyStopping`** — prekida treniranje ako se validaciona tačnost ne poboljšava 10 epoha zaredom, i vraća najbolje težine,
- **`ModelCheckpoint`** — automatski čuva najbolji model (najveća validaciona tačnost) u `models/best_model.keras`.

Trenira se do 60 epoha (ili manje, ako se aktivira `EarlyStopping`). Na kraju se model evaluira na test skupu i ispisuje konačna tačnost, a grafikon tačnosti i greške kroz epohe se čuva kao `training_results.png`.

### 6. Podešavanje hiperparametara — `tune_model.py`

Koristi biblioteku **Keras Tuner** (algoritam `Hyperband`) da automatski isproba više kombinacija hiperparametara modela i pronađe najbolju:

- broj neurona u prvom LSTM sloju (64–256),
- broj neurona u drugom LSTM sloju (32–128),
- stope `Dropout`-a u oba sloja (0.1–0.5),
- broj neurona u `Dense` sloju (32–128),
- learning rate (0.01 / 0.001 / 0.0001).

Najbolja pronađena konfiguracija se ispisuje na kraju, a model sa tim hiperparametrima se čuva u `models/best_params.keras`. Ova skripta se koristi opciono, kada se želi poboljšati tačnost modela iz `train_model.py`.

### 7. Evaluacija modela — `evaluate.py`

Učitava istreniran model (`models/best_model.keras`) i testni skup, pa radi detaljnu analizu performansi:

- ispisuje ukupnu tačnost na test skupu,
- ispisuje `classification_report` (precision, recall, F1 mera za svako slovo),
- izdvaja **slova kod kojih je F1 mera ispod 0.85** kao "problematične" klase,
- za te klase analizira matricu konfuzije i ispisuje **15 najčešćih grešaka** (npr. da model slovo "Š" često zameni za "Ž"),
- generiše grafikon F1 mere po slovu (`class_evaluation.png`), gde su problematična slova označena crvenom bojom.

Ova skripta je korisna da se vidi koja slova model najteže razlikuje (obično ona koja su vizuelno slična).

### 8. Prepoznavanje u realnom vremenu — `realtime.py`

Glavna demonstraciona skripta projekta. Otvara kameru, učitava istreniran model, scaler i listu klasa, i radi prepoznavanje slova "uživo", kroz automat sa četiri stanja:

- **`waiting`** — čeka da se ruka pojavi u kadru.
- **`recording`** — snima 40 frejmova pokreta (isto kao kod `record_samples.py`). Ako ruka izađe iz kadra pre kraja, snimak se odbacuje.
- **`predicting`** — kada se skupi 40 frejmova, sekvenca prolazi kroz **isti pipeline obrade** kao u `prepare_dataset.py` (interpolacija → normalizacija → curl feature-i → skaliranje), a zatim se prosleđuje istreniranom modelu. Model vraća verovatnoće za svako slovo; izdvajaju se **3 najverovatnije opcije (Top 3)** sa procentima, a kao konačna predikcija uzima se slovo sa najvećom verovatnoćom — ali samo ako je ta verovatnoća iznad praga pouzdanosti (40%); inače se koristi znak "?". Ako je prepoznato slovo validno, **automatski se dodaje na trenutnu reč** (bez potrebe za dodatnom potvrdom), nakon čega sistem prelazi u stanje `showing`.
- **`showing`** — prikazuje prepoznato slovo, Top 3 panel i ažuriranu reč na ekranu, i ostaje u ovom stanju sve dok je ruka u kadru (kako se isto slovo ne bi ponovo dodalo). Kada ruka izađe iz kadra, sistem se vraća u `waiting` i čeka novi znak.

Na ekranu se, pored skeletona tela i šaka, prikazuje trenutno stanje, traka napretka snimanja, prepoznato slovo sa trakom pouzdanosti, Top 3 panel sa verovatnoćama, i polje sa rečju koja se sastavlja od automatski prepoznatih slova. Komande tokom rada:

| Taster        | Akcija                       |
| ------------- | ---------------------------- |
| **BACKSPACE** | Briše posledne slovo iz reči |
| **SPACE**     | Briše celu reč               |
| **ESC**       | Izlazak iz programa          |

### Ostale pomoćne skripte

- **`test.py`** — mala skripta za brzu proveru sadržaja jednog `.npy` fajla (oblik, tip podataka, prva vrednost), korisna za debagovanje formata podataka.

## Struktura projekta

```
serbian-sign-language-recognition/
├── data/                       # sirovi podaci (.npy snimci), organizovani po folderima-slovima
├── models/                     # generisani fajlovi: train/val/test skupovi, scaler, istrenirani modeli
├── record_samples.py           # snimanje novih primera preko kamere
├── load_and_explore_data.py    # učitavanje i analiza skupa podataka
├── body_normalization.py       # normalizacija koordinata u odnosu na ramena
├── interpolation.py            # popunjavanje nedostajućih frejmova šake
├── curl_smoothing.py           # izglađivanje feature-a savijenosti prstiju
├── prepare_dataset.py          # kompletna priprema podataka za treniranje
├── train_model.py              # treniranje LSTM modela
├── tune_model.py               # automatsko podešavanje hiperparametara
├── evaluate.py                 # detaljna evaluacija modela po klasama
├── realtime.py                 # prepoznavanje znakova u realnom vremenu
├── test.py                     # provera formata .npy fajla
└── requirements.txt            # spisak Python biblioteka
```

## Uputstvo za pokretanje

### 1. Preduslovi

- Python 3.10 ili 3.11 (TensorFlow 2.16 i MediaPipe 0.10 zahtevaju ove verzije; novije verzije Python-a mogu praviti probleme sa instalacijom).
- Kamera (web kamera), za skripte `record_samples.py` i `realtime.py`.

### 2. Kloniranje repozitorijuma i instalacija zavisnosti

```bash
git clone https://github.com/kajadim/serbian-sign-language-recognition.git
cd serbian-sign-language-recognition

python -m venv venv
source venv/bin/activate        # na Windows-u: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Priprema podataka

Postoje dve opcije:

**Opcija A — koristiti postojeći skup podataka:**
Preuzeti `.npy` snimke iz repozitorijuma [mlradak/serbian-sign-language](https://github.com/mlradak/serbian-sign-language) i smestiti ih u folder `data/`, tako da svako slovo ima svoj podfolder (npr. `data/A/`, `data/B/`, `data/V/`, ...), isto kao u originalnom repozitorijumu.

**Opcija B — snimiti sopstvene primere:**

```bash
python record_samples.py A
```

Ovo pokreće kameru i snima primere za slovo **A** (zameniti sa željenim slovom). Ponoviti za svako slovo koje želimo prepoznavati. Preporučuje se po nekoliko desetina snimaka za svako slovo radi boljeg treniranja.

### 4. (Opciono) Provera skupa podataka

```bash
python load_and_explore_data.py
```

Ispisuje broj klasa, broj snimaka po slovu i čuva grafikon `dataset_analysis.png` sa pregledom skupa podataka.

### 5. Priprema podataka za treniranje

```bash
python prepare_dataset.py
```

Ova skripta obrađuje sve sirove snimke (interpolacija, normalizacija, curl feature-i, skaliranje, podela na trening/validacioni/test skup) i čuva rezultate u folder `models/`.

### 6. Treniranje modela

```bash
python train_model.py
```

Trenira LSTM model i čuva najbolju verziju u `models/best_model.keras`. Na kraju prikazuje tačnost na test skupu i grafikon `training_results.png`.

### 7. (Opciono) Podešavanje hiperparametara

```bash
python tune_model.py
```

Automatski isprobava više konfiguracija modela i ispisuje najbolju pronađenu kombinaciju. Ovaj korak može potrajati duže vreme zavisno od broja kombinacija i veličine skupa podataka.

### 8. Evaluacija modela

```bash
python evaluate.py
```

Prikazuje detaljnu analizu tačnosti po slovima i čuva grafikon `class_evaluation.png`, uz spisak slova koja model najčešće pobrka.

### 9. Prepoznavanje u realnom vremenu

```bash
python realtime.py
```

Otvara kameru i prikazuje prepoznavanje znakova uživo. Postaviti ruku u kadar i izvesti znak za slovo — sistem automatski snima, prepoznaje i dodaje prepoznato slovo na reč, bez potrebe za dodatnom potvrdom. Koristiti **BACKSPACE** za brisanje poslednjeg slova, **SPACE** za reset reči, i **ESC** za izlazak.

> **Napomena:** Pre pokretanja `realtime.py` neophodno je da `models/best_model.keras` i `models/scaler.pkl` već postoje (dobijaju se kroz korake 5 i 6).
