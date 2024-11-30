import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

file_path = 'DadosTreino_Cardiopatas.csv'
data = pd.read_csv(file_path, delimiter=';')

data['2nd_AtaqueCoracao'] = data['2nd_AtaqueCoracao'].map({'Sim': 1, 'Nao': 0})

X = data.drop(columns=['2nd_AtaqueCoracao'])
y = data['2nd_AtaqueCoracao']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

arquiteturas = [
    {'hidden_layer_sizes': (5,), 'activation': 'relu', 'solver': 'adam'},
    {'hidden_layer_sizes': (5,), 'activation': 'relu', 'solver': 'lbfgs'},
    {'hidden_layer_sizes': (5,), 'activation': 'logistic', 'solver': 'adam'},
    {'hidden_layer_sizes': (5,), 'activation': 'logistic', 'solver': 'lbfgs'},
    {'hidden_layer_sizes': (9,), 'activation': 'relu', 'solver': 'adam'},
    {'hidden_layer_sizes': (9,), 'activation': 'relu', 'solver': 'lbfgs'},
    {'hidden_layer_sizes': (9,), 'activation': 'logistic', 'solver': 'adam'},
    {'hidden_layer_sizes': (9,), 'activation': 'logistic', 'solver': 'lbfgs'}
]

resultados = []

for arquitetura in arquiteturas:
    mlp = MLPClassifier(hidden_layer_sizes=arquitetura['hidden_layer_sizes'],
                        activation=arquitetura['activation'],
                        solver=arquitetura['solver'],
                        max_iter=1000,
                        random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    matriz_confusao = confusion_matrix(y_test, y_pred)
    resultados.append({
        'Arquitetura': arquitetura,
        'Acurácia': acuracia,
        'Matriz de Confusão': matriz_confusao
    })

for res in resultados:
    print(f"Arquitetura: {res['Arquitetura']}")
    print(f"Acurácia: {res['Acurácia']:.4f}")
    print(f"Matriz de Confusão:\n{res['Matriz de Confusão']}\n")
