from network.mlp import Language_MLP

english_or_french = Language_MLP(language1='english',language2='german',load_wordlists=False,load_network=True,
                                 save_network=False,balance_wordlists=True,hidden_layers=(20,20))