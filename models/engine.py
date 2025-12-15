import random
from lightning import LightningModule
import numpy as np
from sklearn.metrics import classification_report, precision_recall_curve
from torch import nn
import os
import torch
import matplotlib.pyplot as plt
import wandb
import seaborn as sns
from lion_pytorch import Lion
from torch_ema import ExponentialMovingAverage
from utils.utils_model import pick_model
import constants as cst
from scipy.stats import mode
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LinearLR, SequentialLR



class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, muon_params, adam_params, lr=0.02, weight_decay=0.01, momentum=0.95, 
                 adam_w_lr=0.001, adam_w_betas=(0.9, 0.999), adam_w_eps=1e-8):
        super().__init__(muon_params + adam_params, {})
        self.muon_optim = torch.optim.Muon(muon_params, lr=lr, weight_decay=weight_decay, momentum=momentum)
        self.adam_optim = torch.optim.AdamW(adam_params, lr=adam_w_lr, weight_decay=weight_decay, betas=adam_w_betas, eps=adam_w_eps)
        self.param_groups = self.muon_optim.param_groups + self.adam_optim.param_groups

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        self.muon_optim.step()
        self.adam_optim.step()
        return loss

    def zero_grad(self, set_to_none=False):
        self.muon_optim.zero_grad(set_to_none=set_to_none)
        self.adam_optim.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            'muon': self.muon_optim.state_dict(),
            'adam': self.adam_optim.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.muon_optim.load_state_dict(state_dict['muon'])
        self.adam_optim.load_state_dict(state_dict['adam'])


class Engine(LightningModule):
    def __init__(
        self,
        seq_size,
        horizon,
        max_epochs,
        model_type,
        is_wandb,
        experiment_type,
        lr,
        optimizer,
        dir_ckpt,
        num_features,
        dataset_type,
        num_layers=4,
        num_mlp_layers=None,
        num_trans_layers=None,
        hidden_dim=256,
        num_heads=8,
        is_sin_emb=True,
        len_test_dataloader=None,
        class_weights=None,
        variant="Mlp",
        dropout=0.0,
        muon_lr=0.02,
        muon_momentum=0.95,
        muon_weight_decay=0.01,
        lr_scheduler_type="plateau",
        warmup_epochs=1,
        label_smoothing=0.0,
        use_dyt=False,
        save_attn_score=False,
        use_pos_in_attn=False
    ):
        super().__init__()
        self.save_attn_score = save_attn_score
        self.use_dyt = use_dyt
        self.seq_size = seq_size
        self.dataset_type = dataset_type
        self.horizon = horizon
        self.max_epochs = max_epochs
        self.model_type = model_type
        self.num_heads = num_heads
        self.is_wandb = is_wandb
        self.len_test_dataloader = len_test_dataloader
        self.lr = lr
        self.optimizer = optimizer
        self.dir_ckpt = dir_ckpt
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        self.num_trans_layers = num_trans_layers
        self.num_features = num_features
        self.experiment_type = experiment_type
        self.variant = variant
        self.model = pick_model(model_type, hidden_dim, num_layers, seq_size, num_features, num_heads, is_sin_emb, dataset_type, variant=variant, dropout=dropout, num_mlp_layers=num_mlp_layers, num_trans_layers=num_trans_layers, use_dyt=use_dyt, use_pos_in_attn=use_pos_in_attn)  
        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)
        self.ema.to(cst.DEVICE)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
            weight = self.class_weights
        else:
            weight = None
            self.class_weights = None
        self.loss_function = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.test_targets = []
        self.test_predictions = []
        self.test_proba = []
        self.val_targets = []
        self.val_loss = np.inf
        self.val_predictions = []
        self.min_loss = np.inf
        # class_weights can contain tensors; exclude from hparams to avoid YAML serialization issues
        self.save_hyperparameters(ignore=["class_weights"])
        self.last_path_ckpt = None
        self.first_test = True
        self.test_mid_prices = []
        self.muon_lr = muon_lr
        self.muon_momentum = muon_momentum
        self.muon_weight_decay = muon_weight_decay
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_epochs = warmup_epochs
        self.label_smoothing = label_smoothing
        self.attn_stats = {
            "train": {"sum": [], "count": 0},
            "val": {"sum": [], "count": 0},
            "test": {"sum": [], "count": 0},
        }
        
    def forward(self, x, stock_id=None, batch_idx=None):
        if self.model_type in ("ADALNMLPLOB", "MLPT"):
            output = self.model(x, stock_id)
        else:
            output = self.model(x)
        return output
    
    def loss(self, y_hat, y):
        return self.loss_function(y_hat, y)
        
    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, stock_id = batch
        else:
            x, y = batch
            stock_id = None
            
        if self.model_type == "TLOB" and self.save_attn_score:
            y_hat, att_list = self.model(x, store_att=True)
            self._accumulate_attention(att_list, x.shape[0], mode="train")
        else:
            y_hat = self.forward(x, stock_id, batch_idx)
            
        batch_loss = self.loss(y_hat, y)
        batch_loss_mean = torch.mean(batch_loss)
        self.train_losses.append(batch_loss_mean.item())
        self.ema.update()
        if batch_idx % 1000 == 0:
            print(f'train loss: {sum(self.train_losses) / len(self.train_losses)}')
        return batch_loss_mean
    
    def on_train_epoch_start(self) -> None:
        print(f'learning rate: {self.optimizer.param_groups[0]["lr"]}')

    def on_train_epoch_end(self) -> None:
        self.save_attention_maps("train")
    
    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, stock_id = batch
        else:
            x, y = batch
            stock_id = None
        # Validation: with EMA
        with self.ema.average_parameters():
            if self.model_type == "TLOB" and self.save_attn_score:
                y_hat, att_list = self.model(x, store_att=True)
                self._accumulate_attention(att_list, x.shape[0], mode="val")
            else:
                y_hat = self.forward(x, stock_id, batch_idx)
                
            batch_loss = self.loss(y_hat, y)
            self.val_targets.append(y.cpu().numpy())
            self.val_predictions.append(y_hat.argmax(dim=1).cpu().numpy())
            batch_loss_mean = torch.mean(batch_loss)
            self.val_losses.append(batch_loss_mean.item())
        return batch_loss_mean
        
    
    def test_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, stock_id = batch
        else:
            x, y = batch
            stock_id = None
        mid_prices = ((x[:, 0, 0] + x[:, 0, 2]) // 2).cpu().numpy().flatten()
        self.test_mid_prices.append(mid_prices)
        # Test: with EMA
        if self.experiment_type == "TRAINING":
            with self.ema.average_parameters():
                if self.model_type == "TLOB" and self.save_attn_score:
                    y_hat, att_list = self.model(x, store_att=True)
                    self._accumulate_attention(att_list, x.shape[0], mode="test")
                else:
                    y_hat = self.forward(x, stock_id, batch_idx)
                
                batch_loss = self.loss(y_hat, y)
                self.test_targets.append(y.cpu().numpy())
                self.test_predictions.append(y_hat.argmax(dim=1).cpu().numpy())
                self.test_proba.append(torch.softmax(y_hat, dim=1)[:, 1].cpu().numpy())
                batch_loss_mean = torch.mean(batch_loss)
                self.test_losses.append(batch_loss_mean.item())
        else:
            if self.model_type == "TLOB" and self.save_attn_score:
                y_hat, att_list = self.model(x, store_att=True)
                self._accumulate_attention(att_list, x.shape[0], mode="test")
            else:
                y_hat = self.forward(x, stock_id, batch_idx)
                
            batch_loss = self.loss(y_hat, y)
            self.test_targets.append(y.cpu().numpy())
            self.test_predictions.append(y_hat.argmax(dim=1).cpu().numpy())
            self.test_proba.append(torch.softmax(y_hat, dim=1)[:, 1].cpu().numpy())
            batch_loss_mean = torch.mean(batch_loss)
            self.test_losses.append(batch_loss_mean.item())
        return batch_loss_mean
    
    def on_validation_epoch_start(self) -> None:
        loss = sum(self.train_losses) / len(self.train_losses)
        self.train_losses = []
        # Store train loss for combined plotting
        self.current_train_loss = loss
        print(f'Train loss on epoch {self.current_epoch}: {loss}')
        
    def on_validation_epoch_end(self) -> None:
        self.save_attention_maps("val")
        self.val_loss = sum(self.val_losses) / len(self.val_losses)
        self.val_losses = []
        
        # model checkpointing
        if self.val_loss < self.min_loss:
            self.min_loss = self.val_loss
            self.model_checkpointing(self.val_loss)
        
        # Log losses to wandb (both individually and in the same plot)
        self.log_losses_to_wandb(self.current_train_loss, self.val_loss)
        
        # Continue with regular Lightning logging for compatibility
        self.log("val_loss", self.val_loss)
        print(f'Validation loss on epoch {self.current_epoch}: {self.val_loss}')
        targets = np.concatenate(self.val_targets)    
        predictions = np.concatenate(self.val_predictions)
        class_report = classification_report(targets, predictions, digits=4, output_dict=True)
        print(classification_report(targets, predictions, digits=4))
        self.log("val_f1_score", class_report["macro avg"]["f1-score"])
        self.log("val_accuracy", class_report["accuracy"])
        self.log("val_precision", class_report["macro avg"]["precision"])
        self.log("val_recall", class_report["macro avg"]["recall"])
        self.val_targets = []
        self.val_predictions = [] 
    
    def log_losses_to_wandb(self, train_loss, val_loss):
        """Log training and validation losses to wandb in the same plot."""
        if self.is_wandb:   
            # Log combined losses for a single plot
            wandb.log({
                "losses": {
                    "train": train_loss,
                    "validation": val_loss
                },
                "epoch": self.global_step
            })
    
    def _accumulate_attention(self, att_list, batch_size, mode):
        stats = self.attn_stats[mode]
        if len(stats["sum"]) == 0:
            for att in att_list:
                stats["sum"].append(att.detach().sum(dim=0).cpu())
        else:
            for i, att in enumerate(att_list):
                stats["sum"][i] += att.detach().sum(dim=0).cpu()
        stats["count"] += batch_size

    def save_attention_maps(self, prefix):
        # Map prefix to storage key (test->test, train->train, val->val)
        # Note: 'prefix' argument is used for both directory naming and retrieving stats
        # Ensure prefix matches keys in self.attn_stats
        
        mode = prefix
        stats = self.attn_stats[mode]
        
        if stats["count"] > 0:
            base_dir = os.path.join(cst.DIR_SAVED_MODEL, str(self.model_type), self.dir_ckpt, "attention_maps")
            
            if prefix == "test":
                save_dir = os.path.join(base_dir, "test")
            else:
                save_dir = os.path.join(base_dir, f"epoch_{self.current_epoch}")
                
            os.makedirs(save_dir, exist_ok=True)
            
            for i, attn_sum in enumerate(stats["sum"]):
                # Only process even layers
                if i % 2 != 0:
                    continue
                    
                # Calculate average: (heads, seq, seq)
                avg_attn = attn_sum / stats["count"]
                # Average over heads: (seq, seq)
                avg_attn_heads = avg_attn.mean(dim=0).numpy()
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(avg_attn_heads, cmap="viridis")
                plt.title(f"Layer {i} Average Attention ({prefix})")
                plt.xlabel("Key")
                plt.ylabel("Query")
                
                filename = f"{prefix}_layer_{i}_avg_attn.png"
                plt.savefig(os.path.join(save_dir, filename))
                if self.is_wandb:
                    wandb.log({f"{prefix}_layer_{i}_avg_attn": wandb.Image(plt)})
                plt.close()
            
            # Reset
            stats["sum"] = []
            stats["count"] = 0

    def on_test_epoch_end(self) -> None:
        targets = np.concatenate(self.test_targets)    
        predictions = np.concatenate(self.test_predictions)
        predictions_path = os.path.join(cst.DIR_SAVED_MODEL, str(self.model_type), self.dir_ckpt, "predictions")
        np.save(predictions_path, predictions)
        class_report = classification_report(targets, predictions, digits=4, output_dict=True)
        print(classification_report(targets, predictions, digits=4))
        self.log("test_loss", sum(self.test_losses) / len(self.test_losses))
        self.log("f1_score", class_report["macro avg"]["f1-score"])
        self.log("accuracy", class_report["accuracy"])
        self.log("precision", class_report["macro avg"]["precision"])
        self.log("recall", class_report["macro avg"]["recall"])
        self.test_targets = []
        self.test_predictions = []
        self.test_losses = []  
        self.first_test = False
        test_proba = np.concatenate(self.test_proba)
        precision, recall, _ = precision_recall_curve(targets, test_proba, pos_label=1)
        self.plot_pr_curves(recall, precision, self.is_wandb) 
        
        self.save_attention_maps("test") 
        
    def configure_optimizers(self):
        if self.model_type == "DEEPLOB":
            eps = 1
        else:
            eps = 1e-8
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps=eps)
        elif self.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, eps=eps, weight_decay=0.01)
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer == 'Lion':
            self.optimizer = Lion(self.parameters(), lr=self.lr)
        elif self.optimizer == 'Muon':
            muon_params = []
            adam_params = []
            for name, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                if p.ndim < 2 or "embed" in name.lower() or "final" in name.lower() or "head" in name.lower() or "classifier" in name.lower():
                    adam_params.append(p)
                else:
                    muon_params.append(p)
            print(f"Muon params: {len(muon_params)}, AdamW params: {len(adam_params)}")
            self.optimizer = MuonAdamW(muon_params, adam_params, lr=self.muon_lr, weight_decay=self.muon_weight_decay, momentum=self.muon_momentum, adam_w_lr=self.lr, adam_w_eps=eps)
            
        if self.lr_scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=0, 
                threshold=0.002,
                threshold_mode='abs',
            )
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif self.lr_scheduler_type == "cosine_warmup":
            decay_epochs = self.max_epochs - self.warmup_epochs
            warmup_scheduler = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.warmup_epochs)
            cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=decay_epochs)
            scheduler = SequentialLR(
                self.optimizer, 
                schedulers=[warmup_scheduler, cosine_scheduler], 
                milestones=[self.warmup_epochs]
            )
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
            
        return self.optimizer
    
    def _define_log_metrics(self):
        wandb.define_metric("val_loss", summary="min")

    def model_checkpointing(self, loss):        
        if self.last_path_ckpt is not None:
            os.remove(self.last_path_ckpt)
        filename_ckpt = ("val_loss=" + str(round(loss, 3)) +
                             "_epoch=" + str(self.current_epoch) +
                             ".pt"
                             )
        path_ckpt = os.path.join(cst.DIR_SAVED_MODEL, str(self.model_type), self.dir_ckpt, "pt", filename_ckpt)
        
        # Save PyTorch checkpoint
        with self.ema.average_parameters():
            self.trainer.save_checkpoint(path_ckpt)
        
        self.last_path_ckpt = path_ckpt  
        
    def plot_pr_curves(self, recall, precision, is_wandb):
        plt.figure(figsize=(20, 10), dpi=80)
        plt.plot(recall, precision, label='Precision-Recall', color='black')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        if is_wandb:
            wandb.log({f"precision_recall_curve_{self.dataset_type}": wandb.Image(plt)})
        plt.savefig(cst.DIR_SAVED_MODEL + "/" + str(self.model_type) + "/" +f"precision_recall_curve_{self.dataset_type}.svg")
        #plt.show()
        plt.close()
        
def compute_most_attended(att_feature):
    ''' att_feature: list of tensors of shape (num_samples, num_layers, 2, num_heads, num_features) '''
    att_feature = np.stack(att_feature)
    att_feature = att_feature.transpose(1, 3, 0, 2, 4)  # Use transpose instead of permute
    ''' att_feature: shape (num_layers, num_heads, num_samples, 2, num_features) '''
    indices = att_feature[:, :, :, 1]
    values = att_feature[:, :, :, 0]
    most_frequent_indices = np.zeros((indices.shape[0], indices.shape[1], indices.shape[3]), dtype=int)
    average_values = np.zeros((indices.shape[0], indices.shape[1], indices.shape[3]))
    for layer in range(indices.shape[0]):
        for head in range(indices.shape[1]):
            for seq in range(indices.shape[3]):
                # Extract the indices for the current layer and sequence element
                current_indices = indices[layer, head, :, seq]
                current_values = values[layer, head, :, seq]
                # Find the most frequent index
                most_frequent_index = mode(current_indices, keepdims=False)[0]
                # Store the result
                most_frequent_indices[layer, head, seq] = most_frequent_index
                # Compute the average value for the most frequent index
                avg_value = np.mean(current_values[current_indices == most_frequent_index])
                # Store the average value
                average_values[layer, head, seq] = avg_value
    return most_frequent_indices, average_values



