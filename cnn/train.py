import copy
import gc
import os

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from utils.artifacts import (
    build_session_report_rows,
    maybe_build_roc_curve,
    plot_roc_curves,
    plot_training_curves,
    print_session_report_preview,
    save_checkpoint,
    save_confusion_matrix_artifacts,
    write_run_summary_json,
    write_session_report_csv,
    write_test_only_summary_json,
)
from utils.config import TrainConfig
from utils.data import PreparedData, prepare_data
from utils.metrics import (
    append_epoch_history,
    build_epoch_wandb_payload,
    format_epoch_metrics,
    new_history,
    run_epoch,
)
from utils.model_runtime import (
    amp_enabled,
    build_model,
    build_optimizer,
    build_scheduler,
)
from utils.runtime_utils import (
    configure_torch_matmul,
    get_device,
    seed_everything,
)
from utils.wandb_logging import (
    init_wandb_run,
    update_wandb_run_config,
    wandb_log,
    wandb_log_artifact_images,
    wandb_log_table,
)

class TorchTrainingRun:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.config.validate()
        self.config.ensure_output_dirs()
        seed_everything(self.config.random_state)
        configure_torch_matmul()

        self.device = get_device(cpu_only=self.config.cpu_only)
        self.pin_memory = self.device.type == 'cuda'
        self.criterion = nn.CrossEntropyLoss()
        self.wandb_run: object | None = None
        self.prepared_data: PreparedData | None = None
        self.net: nn.Module | None = None
        self.optimizer: Adam | None = None
        self.scheduler: ReduceLROnPlateau | None = None
        self.scaler: torch.amp.GradScaler | None = None
        self.history = new_history()
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_val_loss = np.inf
        self.best_epoch = 0
        self.resolved_model_artifact_path = self.config.best_model_path

    def print_startup_summary(self) -> None:
        print(f'Using device: {self.device}')
        print(
            f'Training config: input_source={self.config.train_input_source}  '
            f'batch_size={self.config.batch_size}  '
            f'num_workers={self.config.num_workers}  '
            f'amp={"on" if amp_enabled(self.config, self.device) else "off"}'
        )
        if self.config.eval_only:
            print('Evaluation-only mode: enabled')
        if self.config.test_only:
            print('Test-only mode: enabled')
        if self.config.spectrogram_cache_active():
            print(
                'Local spectrogram cache: enabled  '
                f'dir={self.config.spectrogram_cache_dir}'
            )
        else:
            print('Local spectrogram cache: disabled')

    def setup(self) -> None:
        self.print_startup_summary()
        self.wandb_run = init_wandb_run(self.config, self.device)
        self.prepared_data = prepare_data(self.config, self.pin_memory)
        self.net = build_model(self.config, self.device)
        update_wandb_run_config(
            self.wandb_run,
            self.net,
            self.prepared_data.split_summary,
        )
        self.optimizer = build_optimizer(self.net, self.config)
        self.scaler = torch.amp.GradScaler(
            'cuda',
            enabled=amp_enabled(self.config, self.device),
        )
        self.scheduler = build_scheduler(self.optimizer, self.config)

    def run_epoch(
        self,
        loader: DataLoader,
        description: str,
        optimizer: Adam | None = None,
        scaler: torch.amp.GradScaler | None = None,
    ) -> dict[str, np.ndarray | float]:
        if self.net is None:
            raise RuntimeError('Training run has not been set up.')
        return run_epoch(
            self.net,
            loader,
            self.criterion,
            self.config,
            self.device,
            self.pin_memory,
            description,
            optimizer=optimizer,
            scaler=scaler,
        )

    def load_eval_checkpoint(self) -> None:
        if self.net is None:
            raise RuntimeError('Training run has not been set up.')
        checkpoint_path = self.config.eval_checkpoint_path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                'Evaluation-only mode requested but checkpoint does not exist: '
                f'{checkpoint_path}'
            )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.best_epoch = int(checkpoint.get('epoch', 0))
        self.best_val_loss = float(checkpoint.get('val_loss', np.inf))
        self.resolved_model_artifact_path = checkpoint_path
        print(
            f'Loaded checkpoint for evaluation: {checkpoint_path}  '
            f'(epoch={self.best_epoch}, val_loss={self.best_val_loss:.4f})'
        )

    def evaluate_test_split(self) -> None:
        if self.prepared_data is None:
            raise RuntimeError('Training run has not been set up.')
        if self.prepared_data.test_loader is None:
            raise RuntimeError(
                f'Test-only mode requested but split {self.config.test_split!r} '
                'is not available.'
            )

        test_metrics = self.run_epoch(
            self.prepared_data.test_loader,
            description='test final',
        )
        test_session_rows = build_session_report_rows(
            self.config.test_split,
            self.prepared_data.split_datasets[self.config.test_split],
            test_metrics,
        )
        test_session_report_path = write_session_report_csv(
            self.config.test_split,
            test_session_rows,
            self.config,
        )
        (
            test_confusion_csv_path,
            test_confusion_fig_path,
            test_confusion_counts,
        ) = save_confusion_matrix_artifacts(
            self.config.test_split,
            test_metrics,
            self.config,
        )
        test_roc = maybe_build_roc_curve(self.config.test_split, test_metrics)
        plot_roc_curves([test_roc] if test_roc is not None else [], self.config)
        confusion_artifacts = {
            self.config.test_split: {
                'csv_path': test_confusion_csv_path,
                'figure_path': test_confusion_fig_path,
                'counts': test_confusion_counts,
            }
        }
        summary_path = write_test_only_summary_json(
            config=self.config,
            checkpoint_path=self.resolved_model_artifact_path,
            split_summary=self.prepared_data.split_summary,
            test_metrics=test_metrics,
            confusion_artifacts=confusion_artifacts,
            test_session_report_path=test_session_report_path,
        )
        wandb_log_artifact_images(self.wandb_run, self.config)
        wandb_log(
            self.wandb_run,
            {
                'epoch': int(self.best_epoch),
                'test/loss': float(test_metrics['loss']),
                'test/accuracy': float(test_metrics['accuracy']),
                'test/f1': float(test_metrics['f1']),
                'test/precision': float(test_metrics['precision']),
                'test/recall': float(test_metrics['recall']),
                'test/positive_prediction_rate': float(
                    test_metrics['positive_prediction_rate']
                ),
            },
        )
        if test_session_rows:
            wandb_log_table(self.wandb_run, 'test/session_metrics', test_session_rows)
        print('\nTest summary:')
        print(
            f'  Test loss={test_metrics["loss"]:.4f}  '
            f'acc={test_metrics["accuracy"]:.4f}  '
            f'f1={test_metrics["f1"]:.4f}  '
            f'precision={test_metrics["precision"]:.4f}  '
            f'recall={test_metrics["recall"]:.4f}  '
            f'ppr={test_metrics["positive_prediction_rate"]:.4f}'
        )
        print(f'  Checkpoint: {self.resolved_model_artifact_path}')
        print(f'  Run summary: {summary_path}')
        print_session_report_preview(
            self.config.test_split,
            test_session_rows,
            test_session_report_path,
        )

    def train(self) -> None:
        if self.prepared_data is None or self.net is None:
            raise RuntimeError('Training run has not been set up.')
        if self.optimizer is None or self.scheduler is None or self.scaler is None:
            raise RuntimeError('Optimizer, scheduler, or scaler missing.')

        epochs_without_improvement = 0
        try:
            for epoch in range(1, self.config.num_epochs + 1):
                train_metrics = self.run_epoch(
                    self.prepared_data.train_loader,
                    description=f'train epoch {epoch}/{self.config.num_epochs}',
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                )
                val_metrics = self.run_epoch(
                    self.prepared_data.valid_loader,
                    description=f'validation epoch {epoch}/{self.config.num_epochs}',
                )
                self.scheduler.step(float(val_metrics['loss']))

                append_epoch_history(self.history, train_metrics, val_metrics)
                current_lr = float(self.optimizer.param_groups[0]['lr'])
                print(
                    f'Epoch {epoch:02d}/{self.config.num_epochs}  '
                    f'{format_epoch_metrics(train_metrics)}  '
                    f'{format_epoch_metrics(val_metrics, prefix="val_")}  '
                    f'lr={current_lr:.2e}'
                )
                wandb_log(
                    self.wandb_run,
                    build_epoch_wandb_payload(
                        epoch,
                        train_metrics,
                        val_metrics,
                        current_lr,
                        self.best_val_loss,
                    ),
                )

                val_loss = float(val_metrics['loss'])
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.best_state = copy.deepcopy(self.net.state_dict())
                    save_checkpoint(
                        self.config.best_model_path,
                        self.net,
                        self.config,
                        epoch,
                        self.best_val_loss,
                        (
                            self.config.test_split
                            if self.prepared_data.has_test_split
                            else None
                        ),
                    )
                    print(
                        '  -> Best checkpoint saved  '
                        f'(val_loss = {self.best_val_loss:.4f})'
                    )
                    wandb_log(
                        self.wandb_run,
                        {
                            'epoch': epoch,
                            'validation/checkpoint_saved': 1,
                            'validation/best_loss': float(self.best_val_loss),
                            'validation/best_epoch': int(self.best_epoch),
                        },
                    )
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.config.patience:
                    print(f'  -> Early stopping triggered after {epoch} epochs.')
                    break
        except RuntimeError as exc:
            message = str(exc).lower()
            if 'out of memory' in message:
                print(
                    'CUDA OOM during training. Try lowering --batch-size, keep '
                    '--use-amp enabled, and close other GPU-heavy apps.'
                )
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            raise

        if self.best_state is not None:
            self.net.load_state_dict(self.best_state)

    def evaluate_and_write_artifacts(self) -> None:
        if self.prepared_data is None:
            raise RuntimeError('Training run has not been set up.')

        plot_training_curves(self.history, self.config)
        validation_metrics = self.run_epoch(
            self.prepared_data.valid_loader,
            description='validation final',
        )
        test_metrics = (
            self.run_epoch(
                self.prepared_data.test_loader,
                description='test final',
            )
            if self.prepared_data.test_loader is not None
            else None
        )
        validation_session_rows = build_session_report_rows(
            self.config.validation_split,
            self.prepared_data.split_datasets[self.config.validation_split],
            validation_metrics,
        )
        test_session_rows = (
            build_session_report_rows(
                self.config.test_split,
                self.prepared_data.split_datasets[self.config.test_split],
                test_metrics,
            )
            if test_metrics is not None
            else []
        )
        validation_session_report_path = write_session_report_csv(
            self.config.validation_split,
            validation_session_rows,
            self.config,
        )
        test_session_report_path = (
            write_session_report_csv(
                self.config.test_split,
                test_session_rows,
                self.config,
            )
            if test_session_rows
            else None
        )

        validation_confusion = save_confusion_matrix_artifacts(
            self.config.validation_split,
            validation_metrics,
            self.config,
        )
        (
            validation_confusion_csv_path,
            validation_confusion_fig_path,
            validation_confusion_counts,
        ) = validation_confusion
        test_confusion_csv_path = None
        test_confusion_fig_path = None
        test_confusion_counts = None
        if test_metrics is not None:
            (
                test_confusion_csv_path,
                test_confusion_fig_path,
                test_confusion_counts,
            ) = save_confusion_matrix_artifacts(
                self.config.test_split,
                test_metrics,
                self.config,
            )

        roc_curves = []
        validation_roc = maybe_build_roc_curve(
            self.config.validation_split,
            validation_metrics,
        )
        if validation_roc is not None:
            roc_curves.append(validation_roc)
        if test_metrics is not None:
            test_roc = maybe_build_roc_curve(self.config.test_split, test_metrics)
            if test_roc is not None:
                roc_curves.append(test_roc)
        plot_roc_curves(roc_curves, self.config)
        wandb_log_artifact_images(self.wandb_run, self.config)

        confusion_artifacts: dict[str, dict[str, object]] = {
            self.config.validation_split: {
                'csv_path': validation_confusion_csv_path,
                'figure_path': validation_confusion_fig_path,
                'counts': validation_confusion_counts,
            }
        }
        if test_confusion_counts is not None:
            confusion_artifacts[self.config.test_split] = {
                'csv_path': test_confusion_csv_path,
                'figure_path': test_confusion_fig_path,
                'counts': test_confusion_counts,
            }

        run_summary_path = write_run_summary_json(
            config=self.config,
            best_epoch=self.best_epoch,
            best_val_loss=self.best_val_loss,
            best_model_path=self.resolved_model_artifact_path,
            split_summary=self.prepared_data.split_summary,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            confusion_artifacts=confusion_artifacts,
            validation_session_report_path=validation_session_report_path,
            test_session_report_path=test_session_report_path,
        )

        self.print_final_summary(
            validation_metrics,
            test_metrics,
            validation_session_rows,
            test_session_rows,
            validation_session_report_path,
            test_session_report_path,
            run_summary_path,
        )
        self.log_final_wandb(
            validation_metrics,
            test_metrics,
            validation_session_rows,
            test_session_rows,
            validation_session_report_path,
            test_session_report_path,
            run_summary_path,
            validation_confusion_csv_path,
            validation_confusion_fig_path,
            test_confusion_csv_path,
            test_confusion_fig_path,
        )

    def print_final_summary(
        self,
        validation_metrics: dict[str, np.ndarray | float],
        test_metrics: dict[str, np.ndarray | float] | None,
        validation_session_rows: list[dict[str, object]],
        test_session_rows: list[dict[str, object]],
        validation_session_report_path: str | None,
        test_session_report_path: str | None,
        run_summary_path: str,
    ) -> None:
        print('\nFinal summary:')
        print(
            f'  Validation  loss={validation_metrics["loss"]:.4f}  '
            f'acc={validation_metrics["accuracy"]:.4f}  '
            f'f1={validation_metrics["f1"]:.4f}  '
            f'precision={validation_metrics["precision"]:.4f}  '
            f'recall={validation_metrics["recall"]:.4f}  '
            f'ppr={validation_metrics["positive_prediction_rate"]:.4f}'
        )
        if test_metrics is not None:
            print(
                f'  Test        loss={test_metrics["loss"]:.4f}  '
                f'acc={test_metrics["accuracy"]:.4f}  '
                f'f1={test_metrics["f1"]:.4f}  '
                f'precision={test_metrics["precision"]:.4f}  '
                f'recall={test_metrics["recall"]:.4f}  '
                f'ppr={test_metrics["positive_prediction_rate"]:.4f}'
            )
        else:
            print('  Test        skipped (no test split in dataset source)')
        print(f'  Best epoch: {self.best_epoch}')
        print(f'  Best model saved to: {self.resolved_model_artifact_path}')
        print(f'  Run summary: {run_summary_path}')
        print('  Session-level preview:')
        print_session_report_preview(
            self.config.validation_split,
            validation_session_rows,
            validation_session_report_path,
        )
        if test_session_rows:
            print_session_report_preview(
                self.config.test_split,
                test_session_rows,
                test_session_report_path,
            )

    def log_final_wandb(
        self,
        validation_metrics: dict[str, np.ndarray | float],
        test_metrics: dict[str, np.ndarray | float] | None,
        validation_session_rows: list[dict[str, object]],
        test_session_rows: list[dict[str, object]],
        validation_session_report_path: str | None,
        test_session_report_path: str | None,
        run_summary_path: str,
        validation_confusion_csv_path: str,
        validation_confusion_fig_path: str,
        test_confusion_csv_path: str | None,
        test_confusion_fig_path: str | None,
    ) -> None:
        final_wandb_payload = {
            'epoch': int(self.best_epoch),
            'validation/final_loss': float(validation_metrics['loss']),
            'validation/final_accuracy': float(validation_metrics['accuracy']),
            'validation/final_f1': float(validation_metrics['f1']),
            'validation/final_precision': float(validation_metrics['precision']),
            'validation/final_recall': float(validation_metrics['recall']),
            'validation/final_positive_prediction_rate': float(
                validation_metrics['positive_prediction_rate']
            ),
            'training/best_val_loss': float(self.best_val_loss),
            'training/best_epoch': int(self.best_epoch),
            'training/best_model_path': self.resolved_model_artifact_path,
        }
        if test_metrics is not None:
            final_wandb_payload.update(
                {
                    'test/loss': float(test_metrics['loss']),
                    'test/accuracy': float(test_metrics['accuracy']),
                    'test/f1': float(test_metrics['f1']),
                    'test/precision': float(test_metrics['precision']),
                    'test/recall': float(test_metrics['recall']),
                    'test/positive_prediction_rate': float(
                        test_metrics['positive_prediction_rate']
                    ),
                }
            )
        wandb_log(self.wandb_run, final_wandb_payload)
        wandb_log_table(
            self.wandb_run,
            'validation/session_metrics',
            validation_session_rows,
        )
        if test_session_rows:
            wandb_log_table(self.wandb_run, 'test/session_metrics', test_session_rows)
        self.update_wandb_summary(
            validation_metrics,
            test_metrics,
            validation_session_report_path,
            test_session_report_path,
            run_summary_path,
            validation_confusion_csv_path,
            validation_confusion_fig_path,
            test_confusion_csv_path,
            test_confusion_fig_path,
        )

    def update_wandb_summary(
        self,
        validation_metrics: dict[str, np.ndarray | float],
        test_metrics: dict[str, np.ndarray | float] | None,
        validation_session_report_path: str | None,
        test_session_report_path: str | None,
        run_summary_path: str,
        validation_confusion_csv_path: str,
        validation_confusion_fig_path: str,
        test_confusion_csv_path: str | None,
        test_confusion_fig_path: str | None,
    ) -> None:
        if self.wandb_run is None:
            return
        try:
            summary = self.wandb_run.summary
            summary['best_epoch'] = int(self.best_epoch)
            summary['best_val_loss'] = float(self.best_val_loss)
            summary['validation_accuracy'] = float(validation_metrics['accuracy'])
            summary['validation_f1'] = float(validation_metrics['f1'])
            summary['validation_precision'] = float(validation_metrics['precision'])
            summary['validation_recall'] = float(validation_metrics['recall'])
            summary['validation_positive_prediction_rate'] = float(
                validation_metrics['positive_prediction_rate']
            )
            if test_metrics is not None:
                summary['test_accuracy'] = float(test_metrics['accuracy'])
                summary['test_f1'] = float(test_metrics['f1'])
                summary['test_precision'] = float(test_metrics['precision'])
                summary['test_recall'] = float(test_metrics['recall'])
                summary['test_positive_prediction_rate'] = float(
                    test_metrics['positive_prediction_rate']
                )
            if validation_session_report_path is not None:
                summary['validation_session_report_path'] = validation_session_report_path
            if test_session_report_path is not None:
                summary['test_session_report_path'] = test_session_report_path
            summary['run_summary_path'] = run_summary_path
            summary['validation_confusion_csv_path'] = validation_confusion_csv_path
            summary['validation_confusion_fig_path'] = validation_confusion_fig_path
            if test_confusion_csv_path is not None:
                summary['test_confusion_csv_path'] = test_confusion_csv_path
            if test_confusion_fig_path is not None:
                summary['test_confusion_fig_path'] = test_confusion_fig_path
            summary['best_model_path'] = self.resolved_model_artifact_path
        except Exception as exc:
            print(f'Warning: unable to write wandb summary ({exc}).')

    def finish_wandb(self) -> None:
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.finish()
        except Exception as exc:
            print(f'Warning: unable to finish wandb run ({exc}).')

    def cleanup(self) -> None:
        self.prepared_data = None
        self.net = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.best_state = None
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def run(self) -> None:
        try:
            self.setup()
            if self.config.eval_only:
                self.load_eval_checkpoint()
                if self.config.test_only:
                    self.evaluate_test_split()
                    return
            else:
                self.train()
            self.evaluate_and_write_artifacts()
        finally:
            self.finish_wandb()
            self.cleanup()

def main(argv: list[str] | None = None) -> None:
    TorchTrainingRun(TrainConfig.from_args(argv)).run()

if __name__ == '__main__':
    main()
