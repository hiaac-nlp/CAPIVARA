from pytorch_lightning.callbacks.callback import Callback

class AdaptersActivation(Callback):
    """
    """

    def __init__(self, num_layers, activation_finish):
        self.num_layers = num_layers
        self.activation_finish = activation_finish
        self.layer_to_unfreeze = self.num_layers-2
        self.layers_period = self.activation_finish // self.num_layers if self.activation_finish // self.num_layers > 0 else 1

    def on_train_start(self, trainer, pl_module):
        for i in range(self.num_layers-1):
            params_to_train=[f'text_encoder.encoder.layer.{i}.attention.self.query.loras.LoRA.lora_A',
                            f'text_encoder.encoder.layer.{i}.attention.self.query.loras.LoRA.lora_B',
                            f'text_encoder.encoder.layer.{i}.attention.self.value.loras.LoRA.lora_A',
                            f'text_encoder.encoder.layer.{i}.attention.self.value.loras.LoRA.lora_B']
        
            for name, param in pl_module.model.named_parameters():
                if name in params_to_train: param.requires_grad = False

    def on_train_epoch_end(self, trainer, pl_module):
        if (pl_module.current_epoch % self.layers_period) == 0 and self.layer_to_unfreeze >= 0:
            pl_module.model.partial_unfreeze(self.layer_to_unfreeze)
            self.layer_to_unfreeze -= 1