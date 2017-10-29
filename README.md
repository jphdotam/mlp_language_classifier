# mlp_language_classifier
A multilayer perceptron which learns to classify languages from a wordlist. Includes a GUI to visualise neuron activity.

## Requirements / plagiarism ##
- Uses scikit-learn's `MLPClassifier()` class as a network
- Tkinter for visualisation
- Wordlists kindly provided by http://www.gwicks.net/dictionaries.htm (English, French and German supplied)

## Usage ##
- Either run the example scripts e.g. english_to_french_[load or train].py, or:

```
from network.mlp import Language_MLP

english_or_french = Language_MLP(language1='english',language2='french',load_wordlists=False,load_network=True,
                                 save_network=False,balance_wordlists=True,hidden_layers=(20,20))
```
                                 
The above will train a network on words of up to 15 characters using a multi-layer perceptron with a single output neuron. It will train using wordlists, which it will expect in location `"./wordlists/<language1>.txt"` and `"./wordlists/<language1>.txt"`

![Fenetre](https://i.imgur.com/jF7u0fQ.jpg "Fenetre")

![Window](https://i.imgur.com/4BQoGzK.jpg "Window")
