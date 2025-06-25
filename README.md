# AI Algoritmi: Genetska Optimizacija & Rezolucijsko Dokazivanje

Ovaj repozitorij sadrži Python implementacije dvaju algoritama iz područja umjetne inteligencije.

1.  **`nNgenericAlgorithm.py`**: Jednostavna neuronska mreža čije se težine optimiziraju pomoću genetskog algoritma.
2.  **`resolutionAlgorithm.py`**: Rezolucijski dokazivač teorema za propozicijsku logiku.

---

## 1. Genetski algoritam za neuronsku mrežu (`nNgenericAlgorithm.py`)

Ova skripta trenira neuronsku mrežu za regresijske zadatke. Umjesto tradicionalne povratne propagacije, koristi genetski algoritam za pronalaženje optimalnih težina i biasa.

### Pokretanje

Izvršite skriptu iz terminala sa sljedećom strukturom naredbe:

```bash
python nNgenericAlgorithm.py --train podaci_trening.csv --test podaci_test.csv --nn 5s --popsize 50 --elitism 5 --p 0.1 --K 0.05 --iter 20000
```

### Argumenti naredbenog retka

| Argument    | Opis                                                | Primjer      |
|-------------|-------------------------------------------------------------|--------------|
| `--train`   | **Obavezno.** Putanja do CSV datoteke za trening.           | `train.csv`  |
| `--test`    | **Obavezno.** Putanja do CSV datoteke za testiranje.        | `test.csv`   |
| `--nn`      | **Obavezno.** Arhitektura mreže.                            | `5s` ili `5s5s` |
| `--popsize` | **Obavezno.** Broj jedinki u populaciji.                    | `50`         |
| `--elitism` | **Obavezno.** Broj najboljih jedinki koje se prenose dalje. | `5`          |
| `--p`       | **Obavezno.** Vjerojatnost mutacije.                        | `0.1`        |
| `--K`       | **Obavezno.** Standardna devijacija Gaussove mutacije.      | `0.05`       |
| `--iter`    | **Obavezno.** Ukupan broj generacija.                       | `20000`      |

### Format ulazne CSV datoteke

Ulazni podaci trebaju biti u CSV formatu gdje je svaki redak jedan uzorak. Zadnji stupac se tretira kao ciljna vrijednost (`y`), a svi prethodni stupci su ulazne značajke (`X`).

---

## 2. Rezolucijski dokazivač (`resolutionAlgorithm.py`)

Ova skripta implementira princip rezolucije za propozicijsku logiku kako bi dokazala slijedi li zadani zaključak iz skupa klauzula (baze znanja). Ima dva načina rada.

### Način 1: Standardna rezolucija

Ovaj način provjerava može li se ciljna klauzula izvesti iz početnog skupa klauzula.

#### Pokretanje

```bash
python resolutionAlgorithm.py resolution ulaz.txt
```

#### Format ulazne datoteke (`ulaz.txt`)
- Svaka linija predstavlja jednu klauzulu.
- Literali unutar klauzule odvojeni su s ` v `.
- Zadnja linija u datoteci tretira se kao ciljna klauzula koju treba dokazati.
- Linije koje počinju s `#` se ignoriraju.

### Način 2: Interaktivni rad

Ovaj način omogućuje interaktivno dodavanje, uklanjanje ili postavljanje upita nad bazom znanja.

#### Pokretanje

```bash
python resolutionAlgorithm.py cooking baza_znanja.txt naredbe.txt
```

#### Format ulaznih datoteka
- **`baza_znanja.txt`**: Sadrži početni skup klauzula, jedna po retku.
- **`naredbe.txt`**: Sadrži niz naredbi, jedna po retku:
    - **Dodaj klauzulu**: `a v ~c +`
    - **Ukloni klauzulu**: `b v d -`
    - **Postavi upit**: `~a v c ?` 