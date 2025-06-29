import pandas as pd
import matplotlib.pyplot as plt

# CSV-Datei einlesen
df = pd.read_csv(r"C:\Users\hanne\Desktop\Uni\TM2_Team7\Datasets\Modcloth\df_modcloth.csv")  # ← Pfad anpassen

# Datumskonvertierung aus der 'timestamp'-Spalte
df['year'] = pd.to_datetime(df['timestamp'], format='mixed').dt.year


# Durchschnittliches Rating pro Jahr berechnen
average_ratings_per_year = df.groupby('year')['rating'].mean().reset_index()

# Plot erstellen
plt.figure(figsize=(10, 6))
plt.plot(average_ratings_per_year['year'], average_ratings_per_year['rating'], marker='o', linestyle='-')

# Plot beschriften
plt.title("Durchschnittliches Rating pro Jahr")
plt.xlabel("Jahr")
plt.ylabel("Durchschnittliches Rating")
plt.grid(True)
plt.xticks(average_ratings_per_year['year'])  # alle Jahre anzeigen

plt.tight_layout()
plt.show()
