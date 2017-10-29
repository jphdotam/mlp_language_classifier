from network.mlp import Language_MLP

german_or_french = Language_MLP(language1='german',language2='french',load_wordlists=False,load_network=True,
                                 save_network=False,balance_wordlists=True,hidden_layers=(20,20))