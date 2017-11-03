import numpy as np
#import KTimage as kt

GROESSE_HD = 3
GROESSE_INPUT = 4
GROESSE_OUTPUT = 1
ANZAHL_LERNDURCHLAEUFE = 5001
BP_GESCHWINDIGKEIT = 0.1
ANZAHL_DATEN = 30

input = 
for (j in range(ANZAHL_DATEN)):
    
    for (i in range(N)):
        
    




def sigmoid(x):
  return 1 / (1 + np.exp(-x))


input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output = np.array([[0], [0], [0], [1]])

#initialisiert mit Zufallswerten hiddenLayer und outputLayer, sowie Bias

hiddenLayer = np.random.uniform(0., 1. , (GROESSE_HD, GROESSE_INPUT))
outputLayer = np.random.uniform(0., 1., (GROESSE_OUTPUT, GROESSE_HD))

biasHiddenLayer = np.random.uniform(0., 1. , GROESSE_HD)
biasOutputLayer = np.random.uniform(0., 1., GROESSE_OUTPUT)

for durchlauf in range(ANZAHL_LERNDURCHLAEUFE):
    for i in range(len(input)):
        #forward propagation
        
        eingabe = np.copy(input)[i]
        ausgabeHiddenLayer = np.dot(hiddenLayer, eingabe) + biasHiddenLayer # berechnet die Signale des HD
        
        
        sigAusgabeHiddenLayer = sigmoid(ausgabeHiddenLayer)
        sigAbl = sigAusgabeHiddenLayer * (1 - sigAusgabeHiddenLayer) # Ahnma, Ableitung!
        
        AusgabeOutputLayer = np.dot(outputLayer, sigAusgabeHiddenLayer) + biasOutputLayer
        
        #Backward propagation
        fehler = output[i] - AusgabeOutputLayer
        
        backHiddenLayer = sigAbl * np.dot(outputLayer.transpose(), fehler) #backpropragate HD
        
        # Differenzen Gewichte und Biases
        
        deltaOutputLayer = BP_GESCHWINDIGKEIT * np.outer(fehler, sigAusgabeHiddenLayer)
        deltaHiddenLayer = BP_GESCHWINDIGKEIT * np.outer(backHiddenLayer, eingabe)
        
        deltaBiasOutputLayer = BP_GESCHWINDIGKEIT * fehler
        deltaBiasHiddenLayer = BP_GESCHWINDIGKEIT * backHiddenLayer
        
        # Aenderung aller Werte
        
        outputLayer += deltaOutputLayer
        hiddenLayer += deltaHiddenLayer
        
        biasOutputLayer += deltaBiasOutputLayer
        biasHiddenLayer += deltaBiasHiddenLayer
        
        if (durchlauf % 100 == 0):
           print("Durchlauf: " + str(durchlauf))
           print("Koordinate: " + str(i))
           print(fehler)

#kt.exporttiles(array=hiddenLayer, height=1, width=2, outer_height=GROESSE_HD, outer_width=1, filename="results/obs_H_1_0.pgm")






