import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import dual_annealing
from pyswarm import pso
import matplotlib.pyplot as plt

# Definir el modelo LSTM para la predicción de energía
class EnergyLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(EnergyLSTM, self).__init__()
        # Definimos el tamaño de la capa oculta y creamos las capas del modelo
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    # Definimos cómo los datos pasarán a través del modelo
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.linear(hn[-1])
        return out

# Normalización manual de los datos
def normalize_data(df, column_name):
    min_val = df[column_name].min() # Valor mínimo de la columna
    max_val = df[column_name].max()  # Valor máximo de la columna
    # Normalizamos los datos usando la fórmula (dato - min) / (max - min)
    df[column_name] = (df[column_name] - min_val) / (max_val - min_val)
    return df, min_val, max_val 

# Desnormalización de los datos
def denormalize_data(normalized_data, min_val, max_val):
    return normalized_data * (max_val - min_val) + min_val

# Preparar las secuencias de datos
def prepare_sequences(df, sequence_length=24, train_split=0.7):
    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df['FE_MW'].values[i:i + sequence_length])
        y.append(df['FE_MW'].values[i + sequence_length])
    
    # Convertir a arrays de NumPy
    X, y = np.array(X), np.array(y)
    
    # Dividir en entrenamiento y prueba según el porcentaje
    split_index = int(len(X) * train_split)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, y_train, X_test, y_test

# Función para entrenar el modelo
def train_model(X_train, y_train, hidden_layer_size, lr, epochs=10):
    model = EnergyLSTM(input_size=1, hidden_layer_size=hidden_layer_size, output_size=1)
    criterion = nn.MSELoss()  # Usamos MSE como función de pérdida
    optimizer = optim.Adam(model.parameters(), lr=lr) 

    # Convertir datos a tensores de PyTorch
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

    losses = [] 
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(len(X_train)): 
            batch_X = X_train[i:i+1]
            batch_y = y_train[i:i+1]
            
            optimizer.zero_grad() 
            output = model(batch_X) 
            loss = criterion(output, batch_y) 
            loss.backward() 
            optimizer.step() 
            
            epoch_loss += loss.item()

        epoch_loss /= len(X_train)
        losses.append(epoch_loss)

        # Mostrar el gráfico de la pérdida en tiempo real
        plt.clf() 
        plt.plot(losses, label='Pérdida')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.title('Entrenamiento del modelo LSTM')
        plt.legend()
        plt.pause(0.1)  

    return model, losses[-1]  

# Función objetivo para Simulated Annealing
def sa_objective(params, X_train, y_train):
    hidden_layer_size, lr = int(params[0]), params[1]
    _, loss = train_model(X_train, y_train, hidden_layer_size, lr, epochs=5)
    return loss

# Función objetivo para PSO
def pso_objective(params, X_train, y_train):
    hidden_layer_size, lr = int(params[0]), params[1]
    _, loss = train_model(X_train, y_train, hidden_layer_size, lr, epochs=5)
    return loss

def main():
    df = pd.read_csv('FE_hourly_filtered.csv')


    df, min_val, max_val = normalize_data(df, 'FE_MW')
    
    # Preparar las secuencias de datos
    X_train, y_train, X_test, y_test = prepare_sequences(df, train_split=0.7)

    
    bounds = [(10, 100), (0.0001, 0.01)] 


    best_params, best_loss = pso(lambda params: pso_objective(params, X_train, y_train), 
                                    [b[0] for b in bounds], [b[1] for b in bounds], 
                                    swarmsize=20, maxiter=5) 

    print(f"Mejores parámetros con PSO: {best_params}, Mejor pérdida: {best_loss}")


    result_sa = dual_annealing(sa_objective, bounds, args=(X_train, y_train), maxiter=5)
    print(f"Mejores parámetros con SA: {result_sa.x}, Mejor pérdida: {result_sa.fun}")

    # Evaluación del modelo en el conjunto de validación
    best_hidden_layer_size = int(best_params[0])
    best_lr = best_params[1]
    model, _ = train_model(X_train, y_train, best_hidden_layer_size, best_lr, epochs=20)


    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        mse = nn.MSELoss()(predictions, y_test) # Calcula el error cuadrático medio
        print(f"Error cuadrático medio en el conjunto de validación: {mse.item()}")

    # Desnormalizar las predicciones para visualización
    predictions = predictions.numpy().flatten()
    y_test = y_test.numpy().flatten()
    predictions = denormalize_data(predictions, min_val, max_val)
    y_test = denormalize_data(y_test, min_val, max_val)

    # Visualización de los resultados
    plt.plot(y_test, label='Valores Reales')
    plt.plot(predictions, label='Predicciones')
    plt.legend()
    plt.title("Predicciones vs. Valores Reales en el conjunto de validación")
    plt.show()

if __name__ == '__main__':
    main()
