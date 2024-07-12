import torch
from config import Config
config = Config()


class Utils:
    def save(self, model, epoch, file_name)->None:
        checkpoint = {
            "model":model.state_dict(),
            "epoch":epoch,
        }
        torch.save(checkpoint,file_name) 
        print('__finished saving checkpoint__')

        
        
    def load(self, model, file_name)->None: 
        checkpoint = torch.load(file_name, map_location = config.DEVICE)
        model.load_state_dict(checkpoint['model'])
        print('__finished loading checkpoint__')

