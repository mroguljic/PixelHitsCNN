from Trainer import Trainer

#L1U
if True:
    trainer_x = Trainer("config.json","L1U","x")
    trainer_x.train()
    #trainer_x.test()
    #trainer_x.visualize()

    #trainer_y = Trainer("config.json","L1U","x")
    #trainer_y.train()
    #trainer_y.test()
    #trainer_y.visualize()


#L1F
if False:
    trainer_x = Trainer("config.json","L1F","x")
    trainer_x.train()
    trainer_x.test()
    trainer_x.visualize()

    trainer_y = Trainer("config.json","L1F","y")
    trainer_y.train()
    trainer_y.test()
    trainer_y.visualize()

#L2
if False:
    trainer_x = Trainer("config.json","L2","x")
    trainer_x.train()
    trainer_x.test()
    trainer_x.visualize()

    trainer_y = Trainer("config.json","L2","y")
    trainer_y.train()
    trainer_y.test()
    trainer_y.visualize()

if False:
    trainer_x = Trainer("config.json","L2new","x")
    trainer_x.train()
    trainer_x.test()
    trainer_x.visualize()

    trainer_y = Trainer("config.json","L2new","y")
    trainer_y.train()
    trainer_y.test()
    trainer_y.visualize()