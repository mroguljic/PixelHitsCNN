from Trainer import Trainer

for layer in ["L1U","L1F"]:
    trainer_x = Trainer("config.json",layer,"x")
    trainer_x.test()
    trainer_x.visualize_cluster_uncertainty()

    trainer_y = Trainer("config.json",layer,"y")
    trainer_y.test()
    trainer_y.visualize_cluster_uncertainty()