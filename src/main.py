import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("../data/Dados_RH_Turnover.csv", sep=";")

X = data.drop("SaiuDaEmpresa", axis=1)
y = data["SaiuDaEmpresa"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Neural Network": MLPClassifier(max_iter=1000),
}

accuracies = {}
conf_matrices = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies[model_name] = accuracy_score(y_test, y_pred)
    conf_matrices[model_name] = confusion_matrix(y_test, y_pred)

with open("../docs/acuracias_modelos.txt", "w", encoding="utf-8") as f:
    f.write("Acurácia dos modelos:\n")
    for model_name, accuracy in accuracies.items():
        f.write(f"{model_name}: {accuracy:.2f}\n")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Matrizes de Confusão dos Modelos")
axes = axes.flatten()

for idx, (model_name, conf_matrix) in enumerate(conf_matrices.items()):
    sns.heatmap(
        conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx]
    )
    axes[idx].set_title(model_name)
    axes[idx].set_xlabel("Predito")
    axes[idx].set_ylabel("Verdadeiro")

plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.savefig("../docs/matrizes_de_confusao.png")
print("Script executado")
