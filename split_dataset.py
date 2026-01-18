# =============================
# PODZIAŁ ZBIORU DANYCH NA TRAIN / VAL
# =============================

import os
import shutil
import random

# =============================
# 1. ŚCIEŻKI ŹRÓDŁOWE
# =============================
katalog_obrazow = "images"    # folder z wszystkimi obrazami
katalog_etykiet = "labels"   # folder z plikami .txt (YOLO)

# =============================
# 2. ŚCIEŻKI DOCELOWE
# =============================
train_images = "dataset/train/images"
train_labels = "dataset/train/labels"
val_images = "dataset/val/images"
val_labels = "dataset/val/labels"

# =============================
# 3. WYCZYSZCZENIE STARYCH DANYCH
# =============================
shutil.rmtree("dataset/train", ignore_errors=True)
shutil.rmtree("dataset/val", ignore_errors=True)

os.makedirs(train_images, exist_ok=True)
os.makedirs(train_labels, exist_ok=True)
os.makedirs(val_images, exist_ok=True)
os.makedirs(val_labels, exist_ok=True)

# =============================
# 4. WCZYTANIE LISTY OBRAZÓW
# =============================
dozwolone_rozszerzenia = (".jpg", ".jpeg", ".png")

wszystkie_pliki = [
    plik for plik in os.listdir(katalog_obrazow)
    if plik.lower().endswith(dozwolone_rozszerzenia)
]

random.shuffle(wszystkie_pliki)

# =============================
# 5. PODZIAŁ 80% / 20%
# =============================
liczba_train = int(0.8 * len(wszystkie_pliki))

pliki_train = wszystkie_pliki[:liczba_train]
pliki_val = wszystkie_pliki[liczba_train:]

# =============================
# 6. KOPIOWANIE PLIKÓW TRAIN
# =============================
for nazwa_pliku in pliki_train:
    shutil.copy(
        os.path.join(katalog_obrazow, nazwa_pliku),
        train_images
    )

    sciezka_etykiety = os.path.join(
        katalog_etykiet,
        os.path.splitext(nazwa_pliku)[0] + ".txt"
    )

    if os.path.exists(sciezka_etykiety):
        shutil.copy(sciezka_etykiety, train_labels)
    else:
        # jeśli brak etykiety – tworzymy pusty plik
        open(
            os.path.join(train_labels,
                         os.path.splitext(nazwa_pliku)[0] + ".txt"),
            "w"
        ).close()

# =============================
# 7. KOPIOWANIE PLIKÓW VAL
# =============================
for nazwa_pliku in pliki_val:
    shutil.copy(
        os.path.join(katalog_obrazow, nazwa_pliku),
        val_images
    )

    sciezka_etykiety = os.path.join(
        katalog_etykiet,
        os.path.splitext(nazwa_pliku)[0] + ".txt"
    )

    if os.path.exists(sciezka_etykiety):
        shutil.copy(sciezka_etykiety, val_labels)
    else:
        open(
            os.path.join(val_labels,
                         os.path.splitext(nazwa_pliku)[0] + ".txt"),
            "w"
        ).close()

# =============================
# 8. PODSUMOWANIE
# =============================
print("Podział zbioru zakończony")
print(f"Liczba obrazów TRAIN: {len(pliki_train)}")
print(f"Liczba obrazów VAL:   {len(pliki_val)}")
