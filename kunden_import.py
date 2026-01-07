import pandas as pd

path = r"C:\Users\annek\Downloads\NopOpenCart (19).csv"

# Versuch 1: Standard (Komma)
try:
    df = pd.read_csv(path)
except Exception:
    # Versuch 2: falls die Datei Semikolon als Trennzeichen nutzt
    df = pd.read_csv(path, sep=";")

# Attribute (Spalten) + Datentypen
print("\n=== ATTRIBUTE (Spalten) ===")
print(list(df.columns))

print("\n=== DATENTYPEN ===")
print(df.dtypes)

# Erste und letzte 10 Einträge
print("\n=== ERSTE 10 ZEILEN ===")
print(df.head(10))

print("\n=== LETZTE 10 ZEILEN ===")
print(df.tail(10))
