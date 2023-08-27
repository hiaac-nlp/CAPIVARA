from pytorch_lightning.callbacks.callback import Callback


class AdaptersActivation(Callback):
    """
    Progressive unfreezing
    """

    def __init__(self, num_layers, activation_finish):
        self.num_layers = num_layers
        self.activation_finish = activation_finish
        self.layer_to_unfreeze = self.num_layers-2
        self.layers_period = self.activation_finish // self.num_layers if self.activation_finish // self.num_layers > 0 else 1

    def on_train_start(self, trainer, pl_module):
        if self.activation_finish != -1:
            print('------------------ Initiating callback (Progressive unfreezing) -----------------')
            for i in range(self.num_layers-1):
                params_to_train=[f'model.model.text.roberta.encoder.layer.{i}.attention.self.query.loras.LoRA.lora_A',
                                f'model.model.text.roberta.encoder.layer.{i}.attention.self.query.loras.LoRA.lora_B',
                                f'model.model.text.roberta.encoder.layer.{i}.attention.self.value.loras.LoRA.lora_A',
                                f'model.model.text.roberta.encoder.layer.{i}.attention.self.value.loras.LoRA.lora_B']

                for name, param in trainer.model.named_parameters():
                    if name in params_to_train: 
                        param.requires_grad = False
                        print(f'Freezing {name}')
                    # else:
                    #     print(f'{name} dont finded')
                    
    def on_train_epoch_end(self, trainer, pl_module):
        if self.activation_finish != -1:
            if (pl_module.current_epoch % self.layers_period) == 0 and self.layer_to_unfreeze >= 0:
                self.partial_unfreeze(trainer.model)
                self.layer_to_unfreeze -= 1
    
    def partial_unfreeze(self,model):
        params_to_train=[f'model.model.text.roberta.encoder.layer.{self.layer_to_unfreeze}.attention.self.query.loras.LoRA.lora_A',
                        f'model.model.text.roberta.encoder.layer.{self.layer_to_unfreeze}.attention.self.query.loras.LoRA.lora_B',
                        f'model.model.text.roberta.encoder.layer.{self.layer_to_unfreeze}.attention.self.value.loras.LoRA.lora_A',
                        f'model.model.text.roberta.encoder.layer.{self.layer_to_unfreeze}.attention.self.value.loras.LoRA.lora_B']
        
        for name, param in model.named_parameters():
            if name in params_to_train: 
                print(f'--------------------- Unfreezing {name} ---------------------')
                param.requires_grad = True