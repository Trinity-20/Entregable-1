import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# 1. Cargar Datos
df = pd.read_csv("C:/Users/Patrick/Desktop/programacion/entregable/netflix_movies_detailed_up_to_2025.csv")
print(df.columns)
print(df.head())

# Renombrar columnas a español
df.rename(columns={"budget": "Presupuesto", "revenue": "Ingresos", "vote_average": "Calificacion_Promedio"}, inplace=True)

# 2. Preprocesamiento de Datos
df.fillna(method="ffill", inplace=True)
scaler = StandardScaler()
columnas_numericas = ["Presupuesto", "Ingresos", "Calificacion_Promedio"]
df[columnas_numericas] = scaler.fit_transform(df[columnas_numericas])

# Eliminar columnas no numéricas antes de entrenar el modelo
X = df[columnas_numericas]
y = df["Ingresos"]

# 3. Modelo de Machine Learning (Regresión)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)
y_pred = modelo_rf.predict(X_test)
print("Error Absoluto Medio:", mean_absolute_error(y_test, y_pred))

# 4. Modelo de Machine Learning con PyTorch
X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

class RedNeuronal(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(len(columnas_numericas), 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

modelo_torch = RedNeuronal()
criterio = nn.MSELoss()
optimizador = optim.Adam(modelo_torch.parameters(), lr=0.01)

for epoch in range(100):
    optimizador.zero_grad()
    predicciones = modelo_torch(X_tensor)
    perdida = criterio(predicciones, y_tensor)
    perdida.backward()
    optimizador.step()
print("Entrenamiento con PyTorch Completado")

# 5. Procesamiento de Lenguaje Natural (NLP)
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
texto = "Las películas con mayor presupuesto suelen tener mejor rendimiento."
sentimiento = sia.polarity_scores(texto)
print("Análisis de Sentimiento:", sentimiento)

# 6. Visualización de Datos
sns.set_theme(style="whitegrid", palette="pastel")

# 📊 Gráfico de tendencia de ingresos
plt.figure(figsize=(12, 6))
# Gráfico de ingresos
plt.plot(df.index, df["Ingresos"], label="Ingresos", color="#7d3c98", linewidth=2, linestyle="--", marker="o", markersize=5, alpha=0.8)
# Etiquetas mejoradas
plt.xlabel("Índice", fontsize=14, fontweight="bold")
plt.ylabel("Ingresos", fontsize=14, fontweight="bold")
plt.title("📈 Tendencia de Ingresos en Películas", fontsize=16, fontweight="bold", color="#4a235a")
# Separadores de miles en el eje Y
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
# Cuadrícula más suave
plt.grid(True, linestyle="--", alpha=0.5)
# Leyenda bien posicionada
plt.legend(fontsize=12, loc="upper left")
plt.tight_layout()
plt.show()

                                   # 🔥 Matriz de correlación con mejoras en el diseño
plt.figure(figsize=(10, 8))  # Aumentamos el tamaño de la figura

# Mapa de calor con mejor contraste y mayor separación entre valores
sns.heatmap(df[columnas_numericas].corr(), 
            annot=True, 
            cmap="coolwarm", 
            fmt=".2f", 
            linewidths=2,  # Líneas más gruesas para separar las celdas
            square=True, 
            cbar_kws={'shrink': 0.8},  # Barra de color más visible
            annot_kws={"size": 12, "weight": "bold"})  # Agrandamos los valores en el mapa de calor

# Personalización de los textos
plt.title("🔍 Matriz de Correlación", fontsize=18, fontweight="bold", color="#4a235a")
plt.xticks(rotation=45, fontsize=14, color="#2e4053", fontweight="bold")
plt.yticks(fontsize=14, color="#2e4053", fontweight="bold")
# Agregar una nota explicativa dentro del gráfico


plt.tight_layout()
plt.show()


# 📊 Correlación de variables con los ingresos
# Calcular correlaciones con "Ingresos"
correlaciones = df[columnas_numericas].corr()["Ingresos"].drop("Ingresos")
plt.figure(figsize=(9, 6))

# Barras con colores personalizados
correlaciones.plot(kind='bar', 
                   color=sns.color_palette("coolwarm", len(correlaciones)), 
                   alpha=0.85, 
                   edgecolor="black", 
                   width=0.1)  # 🔹 Hace las barras más finas

plt.title("📊 Correlación de Variables con los Ingresos", fontsize=16, fontweight="bold", color="#4a235a")
plt.xlabel("Variables", fontsize=14, fontweight="bold")
plt.ylabel("Coeficiente de Correlación", fontsize=14, fontweight="bold")
plt.xticks(rotation=30, fontsize=12, color="#2e4053")
plt.yticks(fontsize=12, color="#2e4053")

# Líneas de referencia para facilitar lectura
plt.axhline(y=0, color="black", linewidth=1.3, linestyle="--")  # Línea en y=0
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# 7. Generar Informe en PDF
c = canvas.Canvas("informe_final.pdf", pagesize=letter)
c.drawString(100, 750, "Informe Final: Análisis de Datos de Netflix")
c.drawString(100, 730, "Resultados del Modelo de Predicción")
c.save()
print("Informe PDF generado con éxito.")
#python pro.py