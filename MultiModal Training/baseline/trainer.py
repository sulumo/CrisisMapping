import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score
import logging
import numpy as np
import sys

class Trainer:
    def __init__(
        self,
        train_loader,
        dev_loader,
        test_loader,
        model: nn.Module,
        optimizer,
        scheduler,
        save_dir='.',
        display=20,
        device='cuda',
        tensorboard=False,
        task_weights=None,
        eval_mode=False,
        grad_clip_norm=1.0,
    ):
        """
        Multi-task Trainer expecting the model to return a dict:
            outputs = {"task1": logits1, "task2": logits2, "task3": logits3}
        and the dataset to provide integer labels (or -1 for missing):
            data['label_task1'], data['label_task2'], data['label_task3']

        Args:
            train_loader, dev_loader, test_loader: PyTorch DataLoaders
            model: nn.Module (should implement .save(name) if you want model.module.save to work)
            optimizer: optimizer
            scheduler: iteration-based lr scheduler (scheduler.step() called every optimizer step)
            device: 'cuda' or 'cpu'
            task_weights: dict like {'t1':1.0, 't2':1.0, 't3':1.0}
            eval_mode: if True, skip training and only run evaluation (replaces accidental use of built-in eval)
            grad_clip_norm: max norm for gradient clipping (default 1.0); set to None to disable
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.display = display

        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        # FIX Bug 1: was `self.eval = eval` which shadowed Python's built-in eval()
        # and never reflected the --eval CLI flag. Now passed explicitly as eval_mode.
        self.eval_mode = eval_mode
        self.grad_clip_norm = grad_clip_norm
        self.device = device

        self.tensorboard = tensorboard
        if tensorboard:
            self.writer = SummaryWriter()

        # Default task weights
        if task_weights is None:
            task_weights = {
                't1': 0.16,  # Informativeness
                't2': 0.46,  # Humanitarian
                't3': 0.38   # Damage Severity
            } # proportional to square root of the ratio between the no. of classes and the no. of training samples per task
        self.task_weights = task_weights

        # Loss function
        self.ce = nn.CrossEntropyLoss(reduction='mean')

        # AMP scaler (if cuda)
        if self.device != 'cpu':
            self.scaler = torch.cuda.amp.GradScaler()

    # ---------- helper ----------
    @staticmethod
    def _to_device_label_tensor(arr, device):
        # arr can be list/np array/tensor
        if isinstance(arr, torch.Tensor):
            return arr.to(device)
        return torch.tensor(arr, dtype=torch.long, device=device)



    def _safe_compute_task_loss_and_preds(self, logits, labels):
        """
        Compute loss and predictions for a single task with masking.
        Args:
            logits: Tensor [N, C]
            labels: Tensor [N] (with -1 meaning missing)
        Returns:
            loss (scalar tensor) or None if no valid samples
            preds (list) predicted labels on valid indices
            truth (list) true labels on valid indices
            n_valid (int)
        """
        # labels must be torch tensor on same device
        if labels is None:
            return None, [], [], 0

        # Ensure labels are tensor
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long, device=logits.device)

        labels = labels.to(logits.device)

        mask = (labels >= 0)
        if mask.sum().item() == 0:
            return None, [], [], 0

        valid_logits = logits[mask]
        valid_labels = labels[mask]

        loss = self.ce(valid_logits, valid_labels)

        preds = torch.argmax(valid_logits, dim=1).detach().cpu().numpy().tolist()
        truth = valid_labels.detach().cpu().numpy().tolist()

        return loss, preds, truth, mask.sum().item()

    # ---------- training ----------
    def train(self, num_epochs):
        """
        num_epochs: number of training epochs
        """
        best_dev_loss = float('inf')
        patience = 5
        patience_counter = 0

        global_step = 0  # used only for clarity/debugging

        for epoch in range(num_epochs):
            logging.info(f"Training epoch {epoch}")
            self.model.train()

            running_tot_loss = 0.0

            acc_preds = {'t1': [], 't2': [], 't3': []}
            acc_truths = {'t1': [], 't2': [], 't3': []}
            acc_counts = {'t1': 0, 't2': 0, 't3': 0}

            for batch_idx, data in enumerate(tqdm(self.train_loader, total=len(self.train_loader))):
                self.model.train()

                # -------------------------
                # Zero gradients (canonical)
                # -------------------------
                self.optimizer.zero_grad(set_to_none=True)

                # -------------------------
                # Inputs
                # -------------------------
                x = (
                    data['image'].to(self.device),
                    {k: v.to(self.device) for k, v in data['text_tokens'].items()}
                )

                y1 = data.get('label_task1', None)
                y2 = data.get('label_task2', None)
                y3 = data.get('label_task3', None)

                if y1 is not None and not isinstance(y1, torch.Tensor):
                    y1 = torch.tensor(y1, dtype=torch.long, device=self.device)
                if y2 is not None and not isinstance(y2, torch.Tensor):
                    y2 = torch.tensor(y2, dtype=torch.long, device=self.device)
                if y3 is not None and not isinstance(y3, torch.Tensor):
                    y3 = torch.tensor(y3, dtype=torch.long, device=self.device)

                # -------------------------
                # Forward + loss
                # -------------------------
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    outputs = self.model(x)
                    total_loss = 0.0

                    batch_outputs = {}  # store preds/truths for metrics

                    # Task 1
                    if outputs.get('task1') is not None and y1 is not None:
                        l1, p1, t1, n1 = self._safe_compute_task_loss_and_preds(outputs['task1'], y1)
                        if l1 is not None:
                            total_loss += l1 * self.task_weights.get('t1', 1.0)
                            batch_outputs['t1'] = (p1, t1, n1)

                    # Task 2
                    if outputs.get('task2') is not None and y2 is not None:
                        l2, p2, t2, n2 = self._safe_compute_task_loss_and_preds(outputs['task2'], y2)
                        if l2 is not None:
                            total_loss += l2 * self.task_weights.get('t2', 1.0)
                            batch_outputs['t2'] = (p2, t2, n2)

                    # Task 3
                    if outputs.get('task3') is not None and y3 is not None:
                        l3, p3, t3, n3 = self._safe_compute_task_loss_and_preds(outputs['task3'], y3)
                        if l3 is not None:
                            total_loss += l3 * self.task_weights.get('t3', 1.0)
                            batch_outputs['t3'] = (p3, t3, n3)

                # FIX Bug 3: guard the entire backward+scheduler block so that
                # when total_loss==0 the scheduler does NOT advance. Previously
                # `continue` skipped backward/optimizer but the scheduler.step()
                # call sat outside and still ran, drifting the LR curve.
                if total_loss == 0:
                    continue

                # -------------------------
                # Backward + step
                # -------------------------
                if self.scaler is not None:
                    self.scaler.scale(total_loss).backward()
                    # FIX Bug 2: unscale before clipping so grad norms are in true scale
                    if self.grad_clip_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    # FIX Bug 2: gradient clipping for non-AMP path
                    if self.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip_norm
                        )
                    self.optimizer.step()

                # Scheduler advances only when optimizer actually stepped
                self.scheduler.step()
                global_step += 1

                running_tot_loss += float(total_loss.item())

                # -------------------------
                # Accumulate metrics
                # -------------------------
                for tk, (preds, truths, n) in batch_outputs.items():
                    acc_preds[tk].extend(preds)
                    acc_truths[tk].extend(truths)
                    acc_counts[tk] += n

            # -------------------------
            # Epoch summary
            # -------------------------
            logging.info(f"=== Epoch {epoch} finished ===")

            for tk, name in zip(['t1', 't2', 't3'], ['Task1', 'Task2', 'Task3']):
                if acc_counts[tk] > 0:
                    try:
                        micro = f1_score(acc_truths[tk], acc_preds[tk], average='micro')
                        macro = f1_score(acc_truths[tk], acc_preds[tk], average='macro')
                        weighted = f1_score(acc_truths[tk], acc_preds[tk], average='weighted')
                        acc = accuracy_score(acc_truths[tk], acc_preds[tk])
                        logging.info(
                            f"{name} - samples={acc_counts[tk]}, "
                            f"acc={acc:.4f}, micro-F1={micro:.4f}, "
                            f"macro-F1={macro:.4f}, weighted-F1={weighted:.4f}"
                        )
                    except Exception as e:
                        logging.info(f"{name} - metric computation failed: {e}")
                else:
                    logging.info(f"{name} - no labeled samples")

            avg_epoch_loss = running_tot_loss / max(1, len(self.train_loader))
            logging.info(f"Epoch {epoch} avg loss: {avg_epoch_loss:.6f}")

            # -------------------------
            # Validation + early stopping
            # -------------------------
            dev_total_loss = self.validate(epoch)

            if dev_total_loss < best_dev_loss:
                try:
                    if hasattr(self.model, 'module'):
                        self.model.module.save('best')
                    else:
                        self.model.save('best')
                    logging.info("Saved best model.")
                except Exception as e:
                    logging.warning(f"Model save failed: {e}")

                best_dev_loss = dev_total_loss
                patience_counter = 0
            else:
                patience_counter += 1
                logging.info(f"No improvement. Patience {patience_counter}/{patience}")
                if patience_counter >= patience:
                    logging.info("Early stopping triggered.")
                    break


    # ---------- validation ----------
    def validate(self, idx_iter=0):
        """
        Run validation over dev_loader.
        Returns: total_dev_loss = sum of available per-task losses averaged by valid sample counts (or summed)
        """
        self.model.eval()

        # accumulators
        dev_loss_accum = {'t1': 0.0, 't2': 0.0, 't3': 0.0}
        dev_counts = {'t1': 0, 't2': 0, 't3': 0}
        dev_preds = {'t1': [], 't2': [], 't3': []}
        dev_truths = {'t1': [], 't2': [], 't3': []}

        with torch.no_grad():
            for data in tqdm(self.dev_loader, total=len(self.dev_loader)):
                x = (
                    data['image'].to(self.device),
                    {k: v.to(self.device) for k, v in data['text_tokens'].items()}
                )
                y1 = data.get('label_task1', None)
                y2 = data.get('label_task2', None)
                y3 = data.get('label_task3', None)

                if y1 is not None and not isinstance(y1, torch.Tensor):
                    y1 = torch.tensor(y1, dtype=torch.long, device=self.device)
                if y2 is not None and not isinstance(y2, torch.Tensor):
                    y2 = torch.tensor(y2, dtype=torch.long, device=self.device)
                if y3 is not None and not isinstance(y3, torch.Tensor):
                    y3 = torch.tensor(y3, dtype=torch.long, device=self.device)

                outputs = self.model(x)

                # Task1
                if outputs.get('task1', None) is not None and y1 is not None:
                    l1, p1, t1, n1 = self._safe_compute_task_loss_and_preds(outputs['task1'], y1)
                    if l1 is not None:
                        dev_loss_accum['t1'] += float(l1.item()) * n1
                        dev_counts['t1'] += n1
                        dev_preds['t1'].extend(p1)
                        dev_truths['t1'].extend(t1)

                # Task2
                if outputs.get('task2', None) is not None and y2 is not None:
                    l2, p2, t2, n2 = self._safe_compute_task_loss_and_preds(outputs['task2'], y2)
                    if l2 is not None:
                        dev_loss_accum['t2'] += float(l2.item()) * n2
                        dev_counts['t2'] += n2
                        dev_preds['t2'].extend(p2)
                        dev_truths['t2'].extend(t2)

                # Task3
                if outputs.get('task3', None) is not None and y3 is not None:
                    l3, p3, t3, n3 = self._safe_compute_task_loss_and_preds(outputs['task3'], y3)
                    if l3 is not None:
                        dev_loss_accum['t3'] += float(l3.item()) * n3
                        dev_counts['t3'] += n3
                        dev_preds['t3'].extend(p3)
                        dev_truths['t3'].extend(t3)

        # compute averaged losses (per-sample average)
        avg_losses = {}
        for tk in ('t1', 't2', 't3'):
            if dev_counts[tk] > 0:
                avg_losses[tk] = dev_loss_accum[tk] / dev_counts[tk]
            else:
                avg_losses[tk] = 0.0

        # log metrics
        total_dev_loss = 0.0
        logging.info("===== Validation Results =====")
        for tk, name in zip(['t1','t2','t3'], ['Task1','Task2','Task3']):
            if dev_counts[tk] > 0:
                try:
                    micro = f1_score(dev_truths[tk], dev_preds[tk], average='micro')
                    macro = f1_score(dev_truths[tk], dev_preds[tk], average='macro')
                    weighted = f1_score(dev_truths[tk], dev_preds[tk], average='weighted')
                    acc = accuracy_score(dev_truths[tk], dev_preds[tk])
                except Exception:
                    micro = macro = weighted = acc = 0.0

                logging.info(f"{name}: samples={dev_counts[tk]}, loss={avg_losses[tk]:.6f}, acc={acc:.4f}, micro-F1={micro:.4f}, macro-F1={macro:.4f}, weighted-F1={weighted:.4f}")

                # optionally write to tensorboard
                if self.tensorboard:
                    self.writer.add_scalar(f'Val/{name}_loss', avg_losses[tk], idx_iter)
                    self.writer.add_scalar(f'Val/{name}_acc', acc, idx_iter)
                    self.writer.add_scalar(f'Val/{name}_micro_f1', micro, idx_iter)
            else:
                logging.info(f"{name}: no labeled samples in validation")

            total_dev_loss += avg_losses[tk] * self.task_weights.get(tk, 1.0)

        logging.info(f"Total validation loss (weighted sum): {total_dev_loss:.6f}")

        return total_dev_loss

    # ---------- predict/test ----------
    def predict(self):
        """
        Run model on test_loader and produce per-task metrics and per-event reports.
        """
        self.model.eval()
        all_task_preds = {'t1': [], 't2': [], 't3': []}
        all_task_truths = {'t1': [], 't2': [], 't3': []}
        event_labels = {}
        event_predictions = {}

        correct_total = 0
        total_samples = 0

        for data in tqdm(self.test_loader, total=len(self.test_loader)):
            with torch.no_grad():
                x = (
                    data['image'].to(self.device),
                    {k: v.to(self.device) for k, v in data['text_tokens'].items()}
                )
                y1 = data.get('label_task1', None)
                y2 = data.get('label_task2', None)
                y3 = data.get('label_task3', None)

                if y1 is not None and not isinstance(y1, torch.Tensor):
                    y1 = torch.tensor(y1, dtype=torch.long, device=self.device)
                if y2 is not None and not isinstance(y2, torch.Tensor):
                    y2 = torch.tensor(y2, dtype=torch.long, device=self.device)
                if y3 is not None and not isinstance(y3, torch.Tensor):
                    y3 = torch.tensor(y3, dtype=torch.long, device=self.device)

                outputs = self.model(x)

                # Task1
                if outputs.get('task1', None) is not None and y1 is not None:
                    _, p1, t1, n1 = self._safe_compute_task_loss_and_preds(outputs['task1'], y1)
                    all_task_preds['t1'].extend(p1)
                    all_task_truths['t1'].extend(t1)
                    correct_total += sum([int(pp == tt) for pp, tt in zip(p1, t1)])
                    total_samples += n1

                # Task2
                if outputs.get('task2', None) is not None and y2 is not None:
                    _, p2, t2, n2 = self._safe_compute_task_loss_and_preds(outputs['task2'], y2)
                    all_task_preds['t2'].extend(p2)
                    all_task_truths['t2'].extend(t2)
                    correct_total += sum([int(pp == tt) for pp, tt in zip(p2, t2)])
                    total_samples += n2

                # Task3
                if outputs.get('task3', None) is not None and y3 is not None:
                    _, p3, t3, n3 = self._safe_compute_task_loss_and_preds(outputs['task3'], y3)
                    all_task_preds['t3'].extend(p3)
                    all_task_truths['t3'].extend(t3)
                    correct_total += sum([int(pp == tt) for pp, tt in zip(p3, t3)])
                    total_samples += n3

                # gather event-wise for reporting
                events = data.get('event', None)
                if events is not None:
                    # handle event-level grouping (we only record task1 predictions for simplicity)
                    preds_for_event = None
                    # choose which task to report per-event if desired; here we report task1 if present
                    if outputs.get('task1', None) is not None and y1 is not None:
                        _, p1, t1, n1 = self._safe_compute_task_loss_and_preds(outputs['task1'], y1)
                        for i, evt in enumerate(events):
                            evt_key = evt
                            if evt_key not in event_labels:
                                event_labels[evt_key] = []
                                event_predictions[evt_key] = []
                            # find corresponding prediction/truth if available (careful with masking)
                            # We try to match by index: compute mask and index mapping:
                            # For simplicity, we'll skip event aggregation complexity here.

                            # NOTE: you can implement event-level aggregation if needed.

                            pass

        # global accuracy across all valid labeled samples
        test_acc = correct_total / total_samples if total_samples > 0 else 0.0
        logging.info(f"Test overall accuracy (over all available labeled tasks): {test_acc:.4f}")
        
        # per-task metrics
        for tk, name in zip(['t1','t2','t3'], ['Task1','Task2','Task3']):
            preds = all_task_preds[tk]
            truths = all_task_truths[tk]
            if len(truths) > 0:
                try:
                    micro = f1_score(truths, preds, average='micro')
                    macro = f1_score(truths, preds, average='macro')
                    weighted = f1_score(truths, preds, average='weighted')
                    acc = accuracy_score(truths, preds)
                except Exception:
                    micro = macro = weighted = acc = 0.0
                cm = confusion_matrix(truths, preds)
                logging.info(f"{name} - samples={len(truths)}, acc={acc:.4f}, micro-F1={micro:.4f}, macro-F1={macro:.4f}, weighted-F1={weighted:.4f}")
                logging.info(f"{name} confusion matrix:\n{cm}")
            else:
                logging.info(f"{name} - no labeled samples in test set")

        # optionally produce classification report per event (left as optional)
        # for evt, preds in event_predictions.items():
        #    logging.info(classification_report(event_labels[evt], preds))

        return {
            'overall_acc': test_acc,
            'per_task': {
                'task1': {'preds': all_task_preds['t1'], 'truths': all_task_truths['t1']},
                'task2': {'preds': all_task_preds['t2'], 'truths': all_task_truths['t2']},
                'task3': {'preds': all_task_preds['t3'], 'truths': all_task_truths['t3']},
            }
        }
