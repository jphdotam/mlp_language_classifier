from tkinter import *
import time
from wordlist.wordlist import WordList
import unidecode
import numpy as np
from scipy.special import expit as logistic_sigmoid

WIDTH = 1200
HEIGHT = 800
NEURON_DIAMETER = 20
BORDER_LEFT = 100
BORDER_TOP = 100
SPACING_LEFT = 200
SPACING_TOP = 30
LABEL_START_X = 900
LABEL_START_Y = 200
LABEL_SPACING = 50


class MLP_gui:
    '''A gui which accepts an MLPClassifier object and visualises the activity of the neurons'''

    def __init__(self, network, language1, language2, verbose=False):
        self.tk = Tk()
        self.tk.title("MLP")
        self.tk.resizable(0, 0)
        self.tk.wm_attributes("-topmost",1)
        self.language1 = language1
        self.language2 = language2
        self.verbose = verbose

        self.canvas = Canvas(self.tk, width=WIDTH, height=HEIGHT, bd=0, highlightthickness=0)
        self.canvas.pack()
        self.tk.update()

        self.network = network

        self.layers = []
        self.synapses = []

        self.neurons_per_layer = [layer.shape[0] for layer in self.network.coefs_]
        self.neurons_per_layer[0] /= 26

        self.input_neurons = int(self.neurons_per_layer[0])

        self.max_neurons_per_layer = max(self.neurons_per_layer)

        self.activations = []

        def prepare_input(sv):
            word = sv.get()
            if self.verbose: print(word)
            word = unidecode.unidecode(word).lower()
            if (word.isalpha() and len(word) <= self.neurons_per_layer[0]):
                self.run(word)

        sv = StringVar()
        sv.trace("w", lambda name, index, mode, sv=sv: prepare_input(sv))

        self.entry = Entry(self.canvas, textvariable=sv)
        self.entry.pack()
        self.canvas.create_window(0,0,window = self.entry)

        self.letters = []
        self.labels = []

        # Add a layer for each layer
        for layer_id in range(self.network.n_layers_):

            # If final layer, draw outputs
            if layer_id == self.network.n_layers_-1:
                output_neurons = self.network.coefs_[-1].shape[-1]
                self.layers.append(Layer(self, self.canvas, self.network.n_layers_ - 1, output_neurons))

            # Otherwise draw input and hidden layer
            else:
                self.layers.append(Layer(self, self.canvas, layer_id,int(self.neurons_per_layer[layer_id])))

            # When drawing synapses, skip the first layer
            if layer_id == 0: continue

            # Then draw synapses from the layer before to this one
            for neuron1 in self.layers[layer_id-1].neurons:
                for neuron2 in self.layers[layer_id].neurons:
                    self.synapses.append(Synapse(self, self.canvas, neuron1, neuron2))

        while 1:
            self.tk.update_idletasks()
            self.tk.update()
            self.entry.focus_force()
            time.sleep(0.01)

    def run(self, word):
        '''
        1. Draws word in GUI
        2. Calculates activations with forward_propagation()
        3. Draws neurons with draw_neurons()
        '''
        self.draw_word(word)
        one_hot_word = WordList.string_to_onehot(word, self.input_neurons)
        self.activations = self.forward_propagate(one_hot_word)
        self.draw_neurons(self.activations,self.layers,self.canvas)
        self.draw_labels(word,self.activations[-1][0])

    def draw_labels(self,word,activation):
        for label in self.labels:
            self.canvas.delete(label)
        x = LABEL_START_X
        y = LABEL_START_Y
        spacing = LABEL_SPACING

        if activation > 0.5:
            language = self.language2.upper()
            confidence = activation * 100
        else:
            language = self.language1.upper()
            confidence = (1-activation)*100

        labels = [
            ["I think",1],
            ["{}".format(word.upper()),2],
            ["is",1],
            ["{}".format(language),2],
            ["{0:.1f} confident".format(confidence),1]
        ]

        for i,label in enumerate(labels):
            self.labels.append(self.canvas.create_text(x,y+(i*spacing),text=label[0],font=("Avenir",label[1]*20)))

    def draw_neurons(self, activations, layers, canvas):
        '''Draws the neurons on the GUI'''
        if self.verbose: print("Layers: {}, Activations: {}".format(len(layers[1:-1]),len(activations[:-1])))
        for layer,act_layer in zip(layers[1:-1],activations[:-1]):
            activations_layer_normalised = (act_layer - act_layer.min()) / (act_layer.max() - act_layer.min()) * 255
            if self.verbose: print("ALN: {}".format(activations_layer_normalised))
            for neuron,activation in zip(layer.neurons,activations_layer_normalised):
                activation = int(activation)
                r = ("%02x" % 255)
                g = ("%02x" % (255-activation))
                b = ("%02x" % (255-activation))
                color = "#" + r + g + b
                canvas.itemconfig(neuron.id, fill=color)

        language1 = int(activations[-1][0]*255)
        language2 = 255-language1

        r = ("%02x" % 0)
        g = ("%02x" % (255 - language1))
        b = ("%02x" % (255 - language2))
        color = "#" + r + g + b
        canvas.itemconfig(layers[-1].neurons[0].id,fill=color)

    def forward_propagate(self, word):
        activations = []

        input_to_layer = word

        for layer in range(len(self.layers)-2):
            input_to_layer = np.dot(input_to_layer,self.network.coefs_[layer])
            input_to_layer += self.network.intercepts_[layer]
            input_to_layer = np.maximum(input_to_layer, 0)
            activations.append(input_to_layer)

        input_to_layer = np.dot(input_to_layer, self.network.coefs_[-1])
        input_to_layer += self.network.intercepts_[-1]
        input_to_layer = logistic_sigmoid(input_to_layer)
        activations.append(input_to_layer)

        if self.verbose: print("Activations: {}".format(activations))
        word = word.reshape(1,-1)
        if self.verbose: print("Proba from net: {}".format(self.network.predict_proba(word)))
        return activations

    def draw_word(self, word):
        for letter in self.letters:
            self.canvas.delete(letter)
        for i,letter in enumerate(word):
            first_neuron = self.layers[0].neurons[0]
            x = first_neuron.x + NEURON_DIAMETER/2
            y = first_neuron.y + i * SPACING_TOP + NEURON_DIAMETER/2
            self.letters.append(self.canvas.create_text(x,y,text=letter))



class Synapse:
    '''Creates a synapse representing a weight between 2 neurons'''
    def __init__(self, mlp, canvas, neuron1, neuron2):
        self.mlp = mlp
        self.canvas = canvas
        self.neuron1 = neuron1
        self.neuron2 = neuron2

        offset_x_start = NEURON_DIAMETER
        offset_y = NEURON_DIAMETER*0.5

        self.canvas.create_line(self.neuron1.x + offset_x_start, self.neuron1.y + offset_y, self.neuron2.x, self.neuron2.y + offset_y, fill='grey')



class Neuron:
    '''Creates a neuron within a layer (vertical column)
    layer_id reflects columns from left == 0
    neuron_id reflects row from top == 0'''

    def __init__(self, mlp, layer, canvas, neuron_id):
        self.mlp = mlp
        self.layer = layer
        self.canvas = canvas
        self.neuron_id = neuron_id

        self.id = self.canvas.create_oval(0, 0, NEURON_DIAMETER, NEURON_DIAMETER, fill='white')

        self.x = BORDER_LEFT + SPACING_LEFT * self.layer.layer_id
        self.y = BORDER_TOP + ((self.mlp.max_neurons_per_layer - self.layer.neurons_in_layer) * SPACING_TOP/2) + (SPACING_TOP * neuron_id)
        self.canvas.move(self.id, self.x, self.y)



class Layer:
    '''Creates a vertical column of neurones i.e. a layer in the GUI
    The level parameter starts from 0 == input on the left'''

    def __init__(self, mlp, canvas, layer_id, neurons_in_layer):
        self.mlp = mlp
        self.canvas = canvas
        self.layer_id = layer_id
        self.neurons_in_layer = neurons_in_layer

        self.neurons = []
        for neuron in range(neurons_in_layer):
            self.neurons.append(Neuron(self.mlp, self, self.canvas,neuron))