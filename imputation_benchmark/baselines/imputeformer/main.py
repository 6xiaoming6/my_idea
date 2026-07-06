import torch
import argparse
import configparser
from copy import deepcopy
from data import load_data,inverse_Sliding_window
import os
import time
from pypots.imputation import ImputeFormer
from pypots.nn.functional import autocast, calc_mae, calc_rmse
from pypots.utils.logging import logger
import numpy as np
import nni


class ValidationScheduledImputeFormer(ImputeFormer):
    """ImputeFormer with configurable validation cadence only.

    The underlying PyPOTS ImputeFormer module, loss, optimizer, and forward
    pass are untouched. This override mirrors PyPOTS' native training loop and
    only skips validation/early-stop accounting between scheduled epochs.
    """

    def __init__(self, *args, val_epoch=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_epoch = max(1, int(val_epoch))

    def _train_model(self, train_dataloader, val_dataloader=None):
        self.best_model_dict = None
        self.best_loss = float("inf") if self.validation_metric.lower_better else float("-inf")

        try:
            training_step = 0
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                epoch_train_losses = []
                for data in train_dataloader:
                    training_step += 1
                    inputs = self._assemble_input_for_training(data)
                    with autocast(enabled=self.amp_enabled):
                        self.optimizer.zero_grad()
                        results = self.model(inputs, calc_criterion=True)
                        loss = results["loss"].sum()
                        loss.backward()
                        self.optimizer.step()
                    epoch_train_losses.append(loss.item())
                    if self.summary_writer is not None:
                        self._save_log_into_tb_file(training_step, "training", results)

                mean_train_loss = np.mean(epoch_train_losses)
                should_validate = (
                    val_dataloader is not None
                    and (epoch % self.val_epoch == 0 or epoch == self.epochs)
                )
                if should_validate:
                    self.model.eval()
                    val_metrics = []
                    with torch.no_grad():
                        for data in val_dataloader:
                            inputs = self._assemble_input_for_validating(data)
                            with autocast(enabled=self.amp_enabled):
                                results = self.model(inputs, calc_criterion=True)
                            val_metrics.append(results["metric"].sum().detach().item())
                    mean_loss = np.mean(val_metrics)
                    if self.summary_writer is not None:
                        self._save_log_into_tb_file(
                            epoch, "validating", {self.validation_metric_name: mean_loss}
                        )
                    logger.info(
                        f"Epoch {epoch:03d} - training loss ({self.training_loss_name}): "
                        f"{mean_train_loss:.4f}, validation {self.validation_metric_name}: {mean_loss:.4f}"
                    )
                    improved = (
                        self.validation_metric.lower_better and mean_loss < self.best_loss
                    ) or (
                        not self.validation_metric.lower_better and mean_loss > self.best_loss
                    )
                    if improved:
                        self.best_epoch = epoch
                        self.best_loss = mean_loss
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        self.patience = self.original_patience
                    else:
                        self.patience -= 1
                    self._auto_save_model_if_necessary(
                        confirm_saving=self.best_epoch == epoch and self.model_saving_strategy == "better",
                        saving_name=(
                            f"{self.__class__.__name__}_epoch{epoch}_"
                            f"{self.validation_metric_name}{mean_loss:.4f}"
                        ),
                    )
                    if os.getenv("ENABLE_HPO", False):
                        nni.report_intermediate_result(mean_loss)
                    if self.patience == 0:
                        logger.info("Exceeded the training patience. Terminating the training procedure...")
                        break
                elif val_dataloader is None:
                    mean_loss = mean_train_loss
                    logger.info(
                        f"Epoch {epoch:03d} - training loss ({self.training_loss_name}): "
                        f"{mean_train_loss:.4f}"
                    )
                    improved = (
                        self.validation_metric.lower_better and mean_loss < self.best_loss
                    ) or (
                        not self.validation_metric.lower_better and mean_loss > self.best_loss
                    )
                    if improved:
                        self.best_epoch = epoch
                        self.best_loss = mean_loss
                        self.best_model_dict = deepcopy(self.model.state_dict())
                else:
                    logger.info(
                        f"Epoch {epoch:03d} - training loss ({self.training_loss_name}): "
                        f"{mean_train_loss:.4f}, validation skipped (val_epoch={self.val_epoch})"
                    )
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user; loading the best validated checkpoint so far.")

        if self.best_model_dict is None or not np.isfinite(self.best_loss):
            raise ValueError("No finite validation result was produced during ImputeFormer training.")
        logger.info(f"Finished training. The best model is from epoch#{self.best_epoch}.")

def masked_mape_np(y_pred, y_true,  indicating_mask, null_val=np.nan):
    y_pred = np.where(indicating_mask,y_pred,null_val)
    y_true = np.where(indicating_mask,y_true,null_val)
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask & np.greater(y_true,1e-4)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


parser = argparse.ArgumentParser()
parser.add_argument("--data_prefix", default='/mnt/nfsData17/ZhaoMiaomiao1/miss_data/', type=str, help="data file path")
parser.add_argument("--dataset", default='PEMS04', type=str, help="dataset")
parser.add_argument("--logs", default='./logs', type=str, help="log path")
parser.add_argument("--miss_type", default='SR-TR', type=str, help="miss_type")
parser.add_argument("--miss_rate", default=0.5, type=float, help="miss_rate")
parser.add_argument("--val_ratio", default=0.2, type=float, help="val_ratio")
parser.add_argument("--test_ratio", default=0.2, type=float, help="test_ratio")
parser.add_argument("--sample_len", default=12, type=int, help="sample_len")
parser.add_argument("--batch_size", default=32, type=int, help="batch_size")
parser.add_argument("--use_nni", default=0, type=int, help="use_nni")
args = parser.parse_args()


if args.use_nni:
    params = nni.get_next_parameter()
    args.dataset = params['dataset']
    args.miss_type = params['miss_type']
    args.miss_rate = params['miss_rate']

log_root = os.path.join(args.logs,f"{args.dataset}_{args.miss_type}_{args.miss_rate}_{time.ctime()}")
os.mkdir(log_root)

import yaml
model_config_path =  os.path.join("./configurations",f"{args.dataset}.yaml")
with open(model_config_path) as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)
val_epoch = int(model_config.pop('val_epoch', 1))
# model_config['epochs'] = 2
# model_config['patience'] = 1

true_datapath = os.path.join(args.data_prefix,f"{args.dataset}/true_data_{args.miss_type}_{args.miss_rate}_v2.npz")
miss_datapath = os.path.join(args.data_prefix,f"{args.dataset}/miss_data_{args.miss_type}_{args.miss_rate}_v2.npz")

train_set,val_set,test_set,feature_dim,mean , std = load_data(true_datapath,miss_datapath,args.val_ratio,args.test_ratio,args.sample_len)

start = time.time()
model = ValidationScheduledImputeFormer(
    n_steps=args.sample_len, n_features=feature_dim, **model_config,
    batch_size=args.batch_size, saving_path=log_root, device='cuda',
    ORT_weight=0, val_epoch=val_epoch,
)
model.fit(train_set,val_set)
train_time = time.time() - start

# PyPOTS restores ``best_model_dict`` at the end of fit().  Evaluate that best
# checkpoint on the validation set in the original data range; the internal
# validation MSE printed by PyPOTS is on normalized values and is not suitable
# for paper comparison with the other baselines.
val_imputation = model.impute(val_set) * std + mean
val_ori = val_set['X_ori'] * std + mean
val_indicating_mask = np.isnan(val_ori) ^ np.isnan(val_set['X'])
val_mae = calc_mae(val_imputation, np.nan_to_num(val_ori), val_indicating_mask)
val_rmse = calc_rmse(val_imputation, np.nan_to_num(val_ori), val_indicating_mask)
val_mape = masked_mape_np(val_imputation, val_ori, val_indicating_mask)
print(
    f"Validation Metrics Epoch {model.best_epoch}: "
    f"MAE: {val_mae:.6f} RMSE: {val_rmse:.6f} MAPE: {val_mape:.6f}",
    flush=True,
)


start = time.time()
imputation = model.impute(test_set)*std + mean
test_time = time.time() - start

test_ori = test_set['X_ori']*std + mean
indicating_mask = np.isnan(test_ori) ^ np.isnan(test_set['X'])


mae = calc_mae(imputation, np.nan_to_num(test_ori), indicating_mask) 
rmse = calc_rmse(imputation, np.nan_to_num(test_ori), indicating_mask)
mape = masked_mape_np(imputation,test_ori,indicating_mask)

print(log_root)
print(f'{"mae":<12}{"rmse":<12}{"mape":<12}')
print(f'{mae:<12.4f}{rmse:<12.4f}{mape:<12.4f}')
print("GPU memory: ",torch.cuda.max_memory_allocated()/(1024*1024))
print(f"train time: {train_time}\n" + f"test time: {test_time}\n")

imputation = inverse_Sliding_window(imputation)
ground_truth = inverse_Sliding_window(test_ori)

result_npz = os.path.join(log_root,f"result.npz")
np.savez_compressed(result_npz, imputation=imputation, ground_truth=ground_truth)

result_txt = f'{"mae":<12}{"rmse":<12}{"mape":<12}\n'+f'{mae:<12.4f}{rmse:<12.4f}{mape:<12.4f}\n' \
            + f"GPU memory: {torch.cuda.max_memory_allocated()/(1024*1024)}\n"  \
            + f"train time: {train_time}\n" + f"test time: {test_time}\n"

with open(os.path.join(log_root,f"logs.log"), 'w') as f:
    print(result_txt, file=f)

if args.use_nni:
    nni.report_final_result(mae)
