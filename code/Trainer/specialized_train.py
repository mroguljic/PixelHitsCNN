from specialized_Trainer import Trainer

for layer in ["L1U","L1F"]:#,"L3m","L3p","L4m","L4p"]:
    """trainer_x = Trainer("specialized_config.json",layer,"x")
    trainer_x.train()
    trainer_x.test()
    #trainer_x.plot_barycenter_vs_hit("test")
    trainer_x.visualize()"""

    trainer_y = Trainer("specialized_config.json",layer,"y")
    trainer_y.train()
    trainer_y.test()
    #trainer_y.plot_barycenter_vs_hit("test")
    trainer_y.visualize()