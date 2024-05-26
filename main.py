from ultralytics import YOLO
from pytorch_lightning.callbacks import EarlyStopping
import torch
from statistics import mean
# Define the EarlyStopping callback
class EarlyStoppingCallback(EarlyStopping):
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pt', trace_func=None):
        super().__init__(patience=7, verbose=verbose, delta=delta, path=path, trace_func=trace_func)
        self.best_score = float('inf')

    def on_epoch_end(self, trainer, pl_module, outputs, **kwargs):
        val_loss_mean = mean(outputs)
        if val_loss_mean < self.best_score:
            self.best_score = val_loss_mean
            self.save_checkpoint(trainer)
        else:
            if self.patience <= 0:
                # Stop training
                trainer.should_training_stop = True
            else:
                self.patience -= 1

# Initialize the YOLO model
model = YOLO("yolov8n-seg.yaml")

# Set up the EarlyStopping callback
early_stopping_callback = EarlyStoppingCallback(patience=10, verbose=True)

# Train the model with the callback
trainer = torch.jit.TrainLoop(
    model=model,
    train_dataloader=None,  # You need to define your DataLoader here
    val_dataloaders=None,  # And here
    max_epochs=100,
    callbacks=[early_stopping_callback]
)

results = trainer.fit()
