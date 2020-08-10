from  project1.Proj1Trainer import Trainer
from project1.Proj1Net import RNet

trainer = Trainer(RNet,'24',data_path='C:/data',logs_path='D:/data/object1/Rlogs')
trainer.train()