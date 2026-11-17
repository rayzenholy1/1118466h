import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Завдання 1")
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target
print(df.head())
print("Розмір таблиці:", df.shape)

X = df.drop(columns="target")
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("Навчальна вибірка:", X_train.shape)
print("Тестова вибірка:", X_test.shape)

print("\nЗавдання 2")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Частка поясненої дисперсії:", pca.explained_variance_ratio_)
print("Сумарна частка:", pca.explained_variance_ratio_.sum())

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", edgecolor="black")
plt.xlabel("Головна компонента 1")
plt.ylabel("Головна компонента 2")
plt.title("PCA Components Visualization")
plt.legend(handles=scatter.legend_elements()[0], labels=list(wine.target_names), title="Клас")
plt.grid(True)
plt.show()

print("\nЗавдання 3")
components = [2, 5, 10]
accuracies = []
for k in components:
    p = PCA(n_components=k)
    Xtr = p.fit_transform(X_train)
    Xte = p.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(Xtr, y_train)
    acc = accuracy_score(y_test, model.predict(Xte))
    accuracies.append(acc)
    print(f"Компонент: {k}, Accuracy: {acc:.4f}")

print("\nЗавдання 4")
plt.figure(figsize=(8, 5))
plt.plot(components, accuracies, marker="o", color="blue")
plt.xlabel("Кількість компонент")
plt.ylabel("Accuracy")
plt.title("PCA Accuracy vs Components")
plt.xticks(components)
plt.grid(True)
plt.savefig("pca_accuracy_results.png", dpi=150)
plt.show()
print("Графік збережено у файл pca_accuracy_results.png")
