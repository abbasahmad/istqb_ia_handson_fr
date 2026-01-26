import numpy as np
import matplotlib.pyplot as plt

# Table de vérité OR (2 entrées, 1 sortie)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Entrées
Y = np.array([0, 1, 1, 1])                      # Sorties cibles

# Initialisation des poids et biais
W = np.random.rand(2)  # Poids aléatoires [w1, w2]
b = np.random.rand(1)  # Biais aléatoire
#W = [0.9283494469191129, 0.919371310720258]
#b = -0.03469268
print("Les poids : "+str(W)+" Le Biais : "+str(b))
print("################Poids aleatoires " + str(W[0]) + " , " + str(W[1])+ " Biais RANDOM : "+ str(b))

lr = 100           # Taux d'apprentissage
epochs = 1000           # Nombre d'époques

#Reluction step function
def step_function(z):
    return 1 if z >= 0 else 0

# Entraînement du perceptron
for epoch in range(epochs):
    print(f"Époque {epoch + 1}")
    total_error = 0

    for i in range(len(X)):
        # Calcul de la sortie prédite
        z = np.dot(W, X[i]) + b
        Y_pred = step_function(z)
        
        # Calcul de l'erreur
        error = Y[i] - Y_pred
        total_error += abs(error)
        
        # Mise à jour des poids et biais
        W += lr * error * X[i]
        b += lr * error

        print(f"Entrée : {X[i]}, Prédit : {Y_pred}, Cible : {Y[i]}, Erreur : {abs(error)}")
    
    print(f"Erreur totale : {total_error}\n")
    print("################Poids aleatoires " + str(W[0]) + " , " + str(W[1])+ " | Biais RANDOM : "+ str(b))
    if total_error == 0:
        print("Le perceptron a appris la fonction AND.")
        break


# Tester le perceptron
print("Test du modèle entraîné :")
for i in range(len(X)):
    z = np.dot(W, X[i]) + b
    Y_pred = step_function(z)
    print(f"Entrée : {X[i]}, Sortie prédite : {Y_pred}, Cible : {Y[i]}")


# Visualisation
for i in range(len(X)):
    color = 'green' if Y[i] == 1 else 'red'
    plt.scatter(X[i][0], X[i][1], color=color)

x_line = np.linspace(-0.1, 1.1, 100)
y_line = (-W[0] * x_line - b) / W[1]
plt.plot(x_line, y_line, color='blue')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Classification par Perceptron')
#plt.show()