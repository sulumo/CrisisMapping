import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import logging
import numpy as np


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
        eval_mode=False,
        device='cuda',
        tensorboard=False,
        task_weights=None,
        class_weights=None,
        grad_clip_norm=1.0,
    ):
        """
        Multi-task Trainer expecting the model to return a dict:
            outputs = {"task1": logits1, "task2": logits2, "task3": logits3}
        and the dataset to provide integer labels (or -1 for missing):
            data['label_task1'], data['label_task2'], data['label_task3']

        Args:
            train_loader, dev_loader, test_loader : PyTorch DataLoaders
            model       : nn.Module (must implement .save(name) for checkpointing)
            optimizer   : any PyTorch optimizer
            scheduler   : WarmupLinearSchedule  -> stepped per batch
                          ReduceLROnPlateau      -> stepped per epoch with dev loss
                          None                   -> no scheduling
            save_dir    : directory for model checkpoints
            display     : log intermediate metrics every N batches
            eval_mode   : if True, skip training and run evaluation only
            device      : 'cuda' or 'cpu'
            tensorboard : if True, write scalars to SummaryWriter
            task_weights: dict {'t1': float, 't2': float, 't3': float}
                          Weights applied to each task loss in the combined objective.
                          Defaults proportional to sqrt(n_classes / n_train_samples).
            class_weights: dict {'t1': tensor|None, 't2': tensor|None, 't3': tensor|None}
                          Per-class inverse-frequency weights for each task CE loss.
                          Pass None for a task to use uniform (unweighted) CE.
                          Set a class weight to 0.0 to suppress classes never in the data.
                          Example:
                            class_weights={
                                't1': None,
                                't2': torch.tensor([1.0, 1.0, 1.5, 8.0, 3.0, 5.0]),
                                't3': torch.tensor([2.5, 2.0, 1.0]),
                            }
            grad_clip_norm: max gradient norm for clipping (set None to disable)
        """
        self.model      = model
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.save_dir   = save_dir
        self.display    = display
        self.eval_mode  = eval_mode
        self.device     = device
        self.tensorboard = tensorboard
        self.grad_clip_norm = grad_clip_norm

        self.train_loader = train_loader
        self.dev_loader   = dev_loader
        self.test_loader  = test_loader

        if tensorboard:
            self.writer = SummaryWriter()

        # ── Task-level loss weights ───────────────────────────────────
        if task_weights is None:
            task_weights = {
                't1': 0.16,   # Informativeness
                't2': 0.46,   # Humanitarian
                't3': 0.38,   # Damage Severity
            }  # proportional to sqrt(n_classes / n_train_samples) per task
        self.task_weights = task_weights

        # ── Per-task CE loss functions ────────────────────────────────
        # class_weights dict may supply a weight tensor per task or None.
        # nn.CrossEntropyLoss(weight=None) is identical to standard CE,
        # so this interface is fully backward-compatible.
        #
        # NOTE: weights are stored as float32. The AMP-safe forward path
        # casts them to the logit dtype locally at each call — the stored
        # tensors are never mutated.
        if class_weights is None:
            class_weights = {'t1': None, 't2': None, 't3': None}

        self.ce_t1 = nn.CrossEntropyLoss(weight=class_weights.get('t1'), reduction='mean')
        self.ce_t2 = nn.CrossEntropyLoss(weight=class_weights.get('t2'), reduction='mean')
        self.ce_t3 = nn.CrossEntropyLoss(weight=class_weights.get('t3'), reduction='mean')

        for tk, ce in [('t1', self.ce_t1), ('t2', self.ce_t2), ('t3', self.ce_t3)]:
            if ce.weight is not None:
                logging.info(f"Task {tk} class weights: {ce.weight.tolist()}")
            else:
                logging.info(f"Task {tk}: unweighted CE loss")

        # ── AMP scaler ────────────────────────────────────────────────
        # None on CPU — checked before use throughout
        self.scaler = torch.cuda.amp.GradScaler() if self.device != 'cpu' else None

    # ──────────────────────────────────────────────────────────────────
    # Gate weight diagnostic
    # ──────────────────────────────────────────────────────────────────

    def _has_gate(self):
        """Returns True if the underlying model exposes get_gate_weights()."""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        return callable(getattr(model, 'get_gate_weights', None))

    def _log_gate_weights(self, loader, split_name='test'):
        """
        Runs get_gate_weights() over an entire DataLoader split and logs:
          - Global mean gate weight per stage across all samples
          - Per-task, per-class mean gate weights

        Handles two return signatures from get_gate_weights():
          - Option B (shared gate)  : single Tensor (B, 3)
          - Option C (task gates)   : dict {'task1': (B,3), 'task2': (B,3), 'task3': (B,3|4)}

        Task 3 may have 4 stages in Option C+ — handled via _stage_labels().
        Results are logged via logging.info and written to TensorBoard if enabled.
        """
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.eval()

        task_label_keys = {
            't1': 'label_task1',
            't2': 'label_task2',
            't3': 'label_task3',
        }
        task_display_names = {
            't1': 'Task1 (informative)',
            't2': 'Task2 (humanitarian)',
            't3': 'Task3 (damage)',
        }

        # ── Auto-detect return signature with a single probe batch ────
        probe_images = next(iter(loader))['image'][:2].to(self.device)
        with torch.no_grad():
            probe_result = model.get_gate_weights()
        is_task_conditioned = isinstance(probe_result, dict)

        logging.info(
            f"Gate diagnostic mode: "
            f"{'task-conditioned (Option C/C+)' if is_task_conditioned else 'shared (Option B)'}"
        )

        # ── Stage label helper — supports 3-stage and 4-stage gates ──
        stage_names_3 = ['early (layer 3)', 'mid (layer 7)', 'late (pooler)']
        stage_short_3 = ['early', 'mid', 'late']
        stage_names_4 = ['early (layer 3)', 'mid (layer 7)', 'pre-late (layer 10)', 'late (pooler)']
        stage_short_4 = ['early', 'mid', 'prelate', 'late']

        def _stage_labels(n):
            return (stage_names_4, stage_short_4) if n == 4 else (stage_names_3, stage_short_3)

        # ── Accumulators ─────────────────────────────────────────────
        if is_task_conditioned:
            all_gates = {'t1': [], 't2': [], 't3': []}
        else:
            all_gates = []

        task_gates = {'t1': {}, 't2': {}, 't3': {}}  # class_idx -> list of (n_stages,) arrays

        with torch.no_grad():
            for data in tqdm(loader, desc=f'Gate diagnostic [{split_name}]', leave=False):
                images = data['image'].to(self.device)
                raw    = model.get_gate_weights()

                if is_task_conditioned:
                    batch_gates = {
                        't1': raw['task1'].cpu().numpy(),
                        't2': raw['task2'].cpu().numpy(),
                        't3': raw['task3'].cpu().numpy(),
                    }
                    for tk in ('t1', 't2', 't3'):
                        all_gates[tk].append(batch_gates[tk])
                else:
                    batch_gates_arr = raw.cpu().numpy()   # (B, 3)
                    all_gates.append(batch_gates_arr)
                    batch_gates = {tk: batch_gates_arr for tk in ('t1', 't2', 't3')}

                for tk, label_key in task_label_keys.items():
                    labels = data.get(label_key, None)
                    if labels is None:
                        continue
                    labels = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
                    for i, lbl in enumerate(labels):
                        if lbl < 0:
                            continue
                        lbl = int(lbl)
                        task_gates[tk].setdefault(lbl, [])
                        task_gates[tk][lbl].append(batch_gates[tk][i])

        # ── Log global means ─────────────────────────────────────────
        logging.info(f"===== Gate Weight Diagnostic [{split_name}] =====")

        if is_task_conditioned:
            for tk, tname in task_display_names.items():
                concat = np.concatenate(all_gates[tk], axis=0)
                gmean  = concat.mean(axis=0)
                snames, sshort = _stage_labels(len(gmean))
                logging.info(f"Global mean gate weights — {tname}:")
                for s, (sname, w) in enumerate(zip(snames, gmean)):
                    logging.info(f"  Stage {s} — {sname}: {w:.4f}")
                if self.tensorboard:
                    for sname, w in zip(sshort, gmean):
                        self.writer.add_scalar(f'Gates/{split_name}/{tk}/global_{sname}', float(w))
        else:
            concat      = np.concatenate(all_gates, axis=0)
            global_mean = concat.mean(axis=0)
            snames, sshort = _stage_labels(len(global_mean))
            logging.info("Global mean gate weights across all samples:")
            for s, (sname, w) in enumerate(zip(snames, global_mean)):
                logging.info(f"  Stage {s} — {sname}: {w:.4f}")
            if self.tensorboard:
                for sname, w in zip(sshort, global_mean):
                    self.writer.add_scalar(f'Gates/{split_name}/global_{sname}', float(w))

        # ── Log per-task, per-class means ────────────────────────────
        for tk, class_dict in task_gates.items():
            if not class_dict:
                continue
            first_arr    = next(iter(class_dict.values()))[0]
            _, sshort_tk = _stage_labels(len(first_arr))
            logging.info(f"\n  {task_display_names[tk]} — gate weights per class:")
            for class_idx in sorted(class_dict.keys()):
                class_arr  = np.stack(class_dict[class_idx], axis=0)
                class_mean = class_arr.mean(axis=0)
                row = "  ".join([f"{s}: {w:.4f}" for s, w in zip(sshort_tk, class_mean)])
                logging.info(f"    class {class_idx}: {row}  (n={len(class_dict[class_idx])})")
                if self.tensorboard:
                    for sname, w in zip(sshort_tk, class_mean):
                        self.writer.add_scalar(
                            f'Gates/{split_name}/{tk}/class{class_idx}_{sname}', float(w)
                        )

        logging.info("=" * 50)

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_device_label_tensor(arr, device):
        if isinstance(arr, torch.Tensor):
            return arr.to(device)
        return torch.tensor(arr, dtype=torch.long, device=device)

    def _safe_compute_task_loss_and_preds(self, logits, labels, task='t1'):
        """
        Compute masked CE loss and argmax predictions for one task.

        Args:
            logits : Tensor [N, C]
            labels : Tensor [N]  (-1 = missing/unlabeled, masked out)
            task   : 't1' | 't2' | 't3'  — selects the per-task CE loss function

        Returns:
            loss    : scalar Tensor or None (if no valid samples)
            preds   : list[int]  — argmax predictions on valid samples
            truth   : list[int]  — true labels on valid samples
            n_valid : int

        AMP note:
            Under torch.cuda.amp.autocast, logits arrive as float16 while
            the stored ce_fn.weight tensors are float32. Calling the
            nn.CrossEntropyLoss module directly raises a dtype error.
            We bypass it and call F.cross_entropy directly, casting the
            weight to match the logit dtype locally. The stored weight
            is never mutated, preserving float32 for CPU/non-AMP paths.
        """
        if labels is None:
            return None, [], [], 0

        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long, device=logits.device)

        labels = labels.to(logits.device)
        mask   = (labels >= 0)
        if mask.sum().item() == 0:
            return None, [], [], 0

        valid_logits = logits[mask]
        valid_labels = labels[mask]

        ce_fn  = {'t1': self.ce_t1, 't2': self.ce_t2, 't3': self.ce_t3}.get(task, self.ce_t1)
        weight = ce_fn.weight
        if weight is not None:
            weight = weight.to(dtype=valid_logits.dtype, device=valid_logits.device)

        loss  = F.cross_entropy(valid_logits, valid_labels, weight=weight, reduction='mean')
        preds = torch.argmax(valid_logits, dim=1).detach().cpu().numpy().tolist()
        truth = valid_labels.detach().cpu().numpy().tolist()

        return loss, preds, truth, mask.sum().item()

    # ──────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────

    def train(self, num_epochs):
        best_dev_loss    = float('inf')
        patience         = 5
        patience_counter = 0

        for epoch in range(num_epochs):
            logging.info(f"Training epoch {epoch}")
            self.model.train()

            running_tot_loss  = 0.0
            running_cloc_loss = 0.0

            acc_preds  = {'t1': [], 't2': [], 't3': []}
            acc_truths = {'t1': [], 't2': [], 't3': []}
            acc_counts = {'t1': 0,  't2': 0,  't3': 0}

            batch_idx                   = 0
            display_loss_accumulator    = 0.0
            display_samples_accumulator = 0

            for data in tqdm(self.train_loader, total=len(self.train_loader)):
                batch_idx += 1
                self.model.train()

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

                # ── Forward + loss ────────────────────────────────────
                if self.scaler is not None:
                    # AMP path (CUDA)
                    with torch.cuda.amp.autocast():
                        outputs    = self.model(x)
                        loss_terms = {}

                        if outputs.get('task1') is not None and y1 is not None:
                            l1, p1, t1, n1 = self._safe_compute_task_loss_and_preds(outputs['task1'], y1, task='t1')
                            if l1 is not None:
                                loss_terms['t1'] = l1 * self.task_weights.get('t1', 1.0)

                        if outputs.get('task2') is not None and y2 is not None:
                            l2, p2, t2, n2 = self._safe_compute_task_loss_and_preds(outputs['task2'], y2, task='t2')
                            if l2 is not None:
                                loss_terms['t2'] = l2 * self.task_weights.get('t2', 1.0)

                        if outputs.get('task3') is not None and y3 is not None:
                            l3, p3, t3, n3 = self._safe_compute_task_loss_and_preds(outputs['task3'], y3, task='t3')
                            if l3 is not None:
                                loss_terms['t3'] = l3 * self.task_weights.get('t3', 1.0)

                        # Skip batch entirely if no valid labeled samples;
                        # do NOT advance the scheduler on empty batches
                        if len(loss_terms) == 0:
                            continue

                        total_loss = sum(loss_terms.values())

                        # Optional CLOC regularisation term from model output
                        cloc_val = 0.0
                        if 'cloc_loss' in outputs:
                            cloc_val   = float(outputs['cloc_loss'].item())
                            total_loss = total_loss + outputs['cloc_loss']

                    running_tot_loss  += float(total_loss.item())
                    running_cloc_loss += cloc_val

                    # unscale before clipping so norms are in true scale
                    self.scaler.scale(total_loss).backward()
                    if self.grad_clip_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    torch.cuda.empty_cache()

                else:
                    # CPU path
                    outputs    = self.model(x)
                    loss_terms = {}

                    if outputs.get('task1') is not None and y1 is not None:
                        l1, p1, t1, n1 = self._safe_compute_task_loss_and_preds(outputs['task1'], y1, task='t1')
                        if l1 is not None:
                            loss_terms['t1'] = l1 * self.task_weights.get('t1', 1.0)

                    if outputs.get('task2') is not None and y2 is not None:
                        l2, p2, t2, n2 = self._safe_compute_task_loss_and_preds(outputs['task2'], y2, task='t2')
                        if l2 is not None:
                            loss_terms['t2'] = l2 * self.task_weights.get('t2', 1.0)

                    if outputs.get('task3') is not None and y3 is not None:
                        l3, p3, t3, n3 = self._safe_compute_task_loss_and_preds(outputs['task3'], y3, task='t3')
                        if l3 is not None:
                            loss_terms['t3'] = l3 * self.task_weights.get('t3', 1.0)

                    if len(loss_terms) == 0:
                        continue

                    total_loss = sum(loss_terms.values())

                    cloc_val = 0.0
                    if 'cloc_loss' in outputs:
                        cloc_val   = float(outputs['cloc_loss'].item())
                        total_loss = total_loss + outputs['cloc_loss']

                    running_tot_loss  += float(total_loss.item())
                    running_cloc_loss += cloc_val

                    total_loss.backward()
                    if self.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()

                # Per-batch scheduler step (WarmupLinearSchedule only).
                # ReduceLROnPlateau is stepped per-epoch after validation.
                if self.scheduler is not None and not isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step()

                # zero_grad after step — canonical safe position
                self.optimizer.zero_grad(set_to_none=True)

                # ── Accumulate epoch-level metrics ────────────────────
                if outputs.get('task1') is not None and y1 is not None:
                    _, ep1, et1, en1 = self._safe_compute_task_loss_and_preds(outputs['task1'], y1, task='t1')
                    acc_preds['t1'].extend(ep1); acc_truths['t1'].extend(et1); acc_counts['t1'] += en1

                if outputs.get('task2') is not None and y2 is not None:
                    _, ep2, et2, en2 = self._safe_compute_task_loss_and_preds(outputs['task2'], y2, task='t2')
                    acc_preds['t2'].extend(ep2); acc_truths['t2'].extend(et2); acc_counts['t2'] += en2

                if outputs.get('task3') is not None and y3 is not None:
                    _, ep3, et3, en3 = self._safe_compute_task_loss_and_preds(outputs['task3'], y3, task='t3')
                    acc_preds['t3'].extend(ep3); acc_truths['t3'].extend(et3); acc_counts['t3'] += en3

                # ── Mid-epoch display logging ─────────────────────────
                bsz = x[0].shape[0]
                display_loss_accumulator    += float(total_loss.item())
                display_samples_accumulator += bsz

                if batch_idx % self.display == 0:
                    logging.info(f"Epoch {epoch} Batch {batch_idx}:")
                    for tk in ('t1', 't2', 't3'):
                        if acc_counts[tk] > 0:
                            try:
                                micro = f1_score(acc_truths[tk], acc_preds[tk], average='micro')
                                macro = f1_score(acc_truths[tk], acc_preds[tk], average='macro')
                            except Exception:
                                micro = macro = 0.0
                            logging.info(
                                f"  Task {tk}: samples={acc_counts[tk]}, "
                                f"micro-F1={micro:.4f}, macro-F1={macro:.4f}"
                            )
                    logging.info(f"  Running avg loss: {running_tot_loss / max(1, batch_idx):.6f}")

            # ── Epoch summary ─────────────────────────────────────────
            logging.info(f"=== Epoch {epoch} finished ===")
            for tk, name in zip(['t1', 't2', 't3'], ['Task1', 'Task2', 'Task3']):
                if acc_counts[tk] > 0:
                    try:
                        micro    = f1_score(acc_truths[tk], acc_preds[tk], average='micro')
                        macro    = f1_score(acc_truths[tk], acc_preds[tk], average='macro')
                        weighted = f1_score(acc_truths[tk], acc_preds[tk], average='weighted')
                        acc      = accuracy_score(acc_truths[tk], acc_preds[tk])
                        logging.info(
                            f"{name} - samples={acc_counts[tk]}, acc={acc:.4f}, "
                            f"micro-F1={micro:.4f}, macro-F1={macro:.4f}, weighted-F1={weighted:.4f}"
                        )
                    except Exception as e:
                        logging.info(f"{name} - metric computation failed: {e}")
                else:
                    logging.info(f"{name} - no labeled samples")

            avg_epoch_loss = running_tot_loss  / max(1, len(self.train_loader))
            avg_cloc_loss  = running_cloc_loss / max(1, len(self.train_loader))
            logging.info(
                f"Epoch {epoch} avg total loss: {avg_epoch_loss:.6f} "
                f"(CLOC: {avg_cloc_loss:.6f}, task: {avg_epoch_loss - avg_cloc_loss:.6f})"
            )

            # ── Validation + early stopping ───────────────────────────
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
                best_dev_loss    = dev_total_loss
                patience_counter = 0
            else:
                patience_counter += 1
                logging.info(f"No improvement. Patience {patience_counter}/{patience}")
                if patience_counter >= patience:
                    logging.info("Early stopping triggered.")
                    break

            # ReduceLROnPlateau steps per-epoch with the validation loss.
            # WarmupLinearSchedule was already stepped per-batch above.
            try:
                if self.scheduler is not None and isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(dev_total_loss)
                    logging.info(
                        f"ReduceLROnPlateau stepped. "
                        f"Current LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                    )
            except Exception as e:
                logging.debug(f"Scheduler step failed: {e}")

    # ──────────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────────

    def validate(self, idx_iter=0):
        """
        Run one pass over dev_loader.
        Returns the weighted sum of per-task per-sample-averaged losses,
        used as the early-stopping and ReduceLROnPlateau metric.
        """
        self.model.eval()

        dev_loss_accum = {'t1': 0.0, 't2': 0.0, 't3': 0.0}
        dev_counts     = {'t1': 0,   't2': 0,   't3': 0}
        dev_preds      = {'t1': [],  't2': [],  't3': []}
        dev_truths     = {'t1': [],  't2': [],  't3': []}

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

                if outputs.get('task1') is not None and y1 is not None:
                    l1, p1, t1, n1 = self._safe_compute_task_loss_and_preds(outputs['task1'], y1, task='t1')
                    if l1 is not None:
                        dev_loss_accum['t1'] += float(l1.item()) * n1
                        dev_counts['t1'] += n1
                        dev_preds['t1'].extend(p1)
                        dev_truths['t1'].extend(t1)

                if outputs.get('task2') is not None and y2 is not None:
                    l2, p2, t2, n2 = self._safe_compute_task_loss_and_preds(outputs['task2'], y2, task='t2')
                    if l2 is not None:
                        dev_loss_accum['t2'] += float(l2.item()) * n2
                        dev_counts['t2'] += n2
                        dev_preds['t2'].extend(p2)
                        dev_truths['t2'].extend(t2)

                if outputs.get('task3') is not None and y3 is not None:
                    l3, p3, t3, n3 = self._safe_compute_task_loss_and_preds(outputs['task3'], y3, task='t3')
                    if l3 is not None:
                        dev_loss_accum['t3'] += float(l3.item()) * n3
                        dev_counts['t3'] += n3
                        dev_preds['t3'].extend(p3)
                        dev_truths['t3'].extend(t3)

        avg_losses     = {}
        total_dev_loss = 0.0

        logging.info("===== Validation Results =====")
        for tk, name in zip(['t1', 't2', 't3'], ['Task1', 'Task2', 'Task3']):
            avg_losses[tk] = (
                dev_loss_accum[tk] / dev_counts[tk] if dev_counts[tk] > 0 else 0.0
            )
            if dev_counts[tk] > 0:
                try:
                    micro    = f1_score(dev_truths[tk], dev_preds[tk], average='micro')
                    macro    = f1_score(dev_truths[tk], dev_preds[tk], average='macro')
                    weighted = f1_score(dev_truths[tk], dev_preds[tk], average='weighted')
                    acc      = accuracy_score(dev_truths[tk], dev_preds[tk])
                except Exception:
                    micro = macro = weighted = acc = 0.0

                logging.info(
                    f"{name}: samples={dev_counts[tk]}, loss={avg_losses[tk]:.6f}, "
                    f"acc={acc:.4f}, micro-F1={micro:.4f}, macro-F1={macro:.4f}, "
                    f"weighted-F1={weighted:.4f}"
                )
                if self.tensorboard:
                    self.writer.add_scalar(f'Val/{name}_loss',     avg_losses[tk], idx_iter)
                    self.writer.add_scalar(f'Val/{name}_acc',      acc,            idx_iter)
                    self.writer.add_scalar(f'Val/{name}_micro_f1', micro,          idx_iter)
            else:
                logging.info(f"{name}: no labeled samples in validation")

            total_dev_loss += avg_losses[tk] * self.task_weights.get(tk, 1.0)

        logging.info(f"Total validation loss (weighted sum): {total_dev_loss:.6f}")
        return total_dev_loss

    # ──────────────────────────────────────────────────────────────────
    # Predict / test
    # ──────────────────────────────────────────────────────────────────

    def predict(self):
        """
        Run model on test_loader, produce per-task metrics, and — if the model
        exposes get_gate_weights() — automatically log the gate weight diagnostic
        over the test split at the end.
        """
        self.model.eval()

        all_task_preds  = {'t1': [], 't2': [], 't3': []}
        all_task_truths = {'t1': [], 't2': [], 't3': []}
        correct_total   = 0
        total_samples   = 0

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

                if outputs.get('task1') is not None and y1 is not None:
                    _, p1, t1, n1 = self._safe_compute_task_loss_and_preds(outputs['task1'], y1, task='t1')
                    all_task_preds['t1'].extend(p1)
                    all_task_truths['t1'].extend(t1)
                    correct_total += sum(int(pp == tt) for pp, tt in zip(p1, t1))
                    total_samples += n1

                if outputs.get('task2') is not None and y2 is not None:
                    _, p2, t2, n2 = self._safe_compute_task_loss_and_preds(outputs['task2'], y2, task='t2')
                    all_task_preds['t2'].extend(p2)
                    all_task_truths['t2'].extend(t2)
                    correct_total += sum(int(pp == tt) for pp, tt in zip(p2, t2))
                    total_samples += n2

                if outputs.get('task3') is not None and y3 is not None:
                    _, p3, t3, n3 = self._safe_compute_task_loss_and_preds(outputs['task3'], y3, task='t3')
                    all_task_preds['t3'].extend(p3)
                    all_task_truths['t3'].extend(t3)
                    correct_total += sum(int(pp == tt) for pp, tt in zip(p3, t3))
                    total_samples += n3

        test_acc = correct_total / total_samples if total_samples > 0 else 0.0
        logging.info(f"Test overall accuracy (over all available labeled tasks): {test_acc:.4f}")

        for tk, name in zip(['t1', 't2', 't3'], ['Task1', 'Task2', 'Task3']):
            preds  = all_task_preds[tk]
            truths = all_task_truths[tk]
            if len(truths) > 0:
                try:
                    micro    = f1_score(truths, preds, average='micro')
                    macro    = f1_score(truths, preds, average='macro')
                    weighted = f1_score(truths, preds, average='weighted')
                    acc      = accuracy_score(truths, preds)
                except Exception:
                    micro = macro = weighted = acc = 0.0
                cm = confusion_matrix(truths, preds)
                logging.info(
                    f"{name} - samples={len(truths)}, acc={acc:.4f}, "
                    f"micro-F1={micro:.4f}, macro-F1={macro:.4f}, weighted-F1={weighted:.4f}"
                )
                logging.info(f"{name} confusion matrix:\n{cm}")
            else:
                logging.info(f"{name} - no labeled samples in test set")

        # ── Gate weight diagnostic (auto-runs if model supports it) ───
        if self._has_gate():
            logging.info("Model supports gate weights — running diagnostic over test split...")
            self._log_gate_weights(self.test_loader, split_name='test')

        return {
            'overall_acc': test_acc,
            'per_task': {
                'task1': {'preds': all_task_preds['t1'], 'truths': all_task_truths['t1']},
                'task2': {'preds': all_task_preds['t2'], 'truths': all_task_truths['t2']},
                'task3': {'preds': all_task_preds['t3'], 'truths': all_task_truths['t3']},
            }
        }