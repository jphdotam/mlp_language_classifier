from network.mlp import Language_MLP

english_or_french = Language_MLP(language1='english',language2='french',load_wordlists=True,load_network=False,
                                 save_network=True,balance_wordlists=True,hidden_layers=(20,20))