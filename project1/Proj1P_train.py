from project1.Proj1Trainer import Trainer
from project1.Proj1Net import PNet

trainer = Trainer(PNet, '12', data_path='C:/data', logs_path='D:/data/object1/plogs')
trainer.train()
