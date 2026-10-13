"""Training pipeline for WiFi fingerprinting models."""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from ..data.dataset import WiFiFingerprintDataModule
from ..data.transform_factory import build_transform
from ..models.registry import (
    GRAPH_MODELS as _GRAPH_MODELS,
    IMAGE_MODELS as _IMAGE_MODELS,
    accepted_params as _accepted_params,
    get_model_class as _import_model_class,
)


def sparse_transformer_collate_fn(batch):
    """Dense RSSI -> padded (ap_indices, rssi_values, padding_mask) for the variable-length transformer."""
    features_list, targets_list, metadata_list = zip(*batch)

    targets = torch.stack(targets_list)
    metadata = {k: [m[k] for m in metadata_list] for k in metadata_list[0]}

    sparse_samples = []
    for feat in features_list:
        detected = feat.nonzero(as_tuple=True)[0]
        sparse_samples.append((detected, feat[detected]))

    max_len = max(s[0].size(0) for s in sparse_samples)
    max_len = max(max_len, 1)
    batch_size = len(sparse_samples)

    ap_indices = torch.zeros(batch_size, max_len, dtype=torch.long)
    rssi_values = torch.zeros(batch_size, max_len)
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)

    for i, (idx, vals) in enumerate(sparse_samples):
        n = idx.size(0)
        ap_indices[i, :n] = idx
        rssi_values[i, :n] = vals
        padding_mask[i, :n] = False

    features_dict = {
        'ap_indices': ap_indices,
        'rssi_values': rssi_values,
        'padding_mask': padding_mask,
    }
    return features_dict, targets, metadata


class WiFiTrainer:

    def __init__(
        self,
        config: Dict[str, Any],
        model_name: str,
        experiment_dir: str = "experiments",
        resume_from: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        self.config = config
        self.model_name = model_name  # run name, model type comes from runs.<name>.model

        run_cfg = config.get('runs', {}).get(model_name, {})
        if run_cfg and 'model' in run_cfg:
            self.model_type = run_cfg['model']
        else:
            self.model_type = model_name

        self.experiment_dir = Path(experiment_dir)
        self.resume_from = resume_from

        if experiment_name:
            self.run_dir = self.experiment_dir / model_name / experiment_name
        elif resume_from:
            self.run_dir = self._get_experiment_dir_from_checkpoint(resume_from)
        else:
            self.run_dir = self._generate_experiment_name()

        self.run_dir.mkdir(parents=True, exist_ok=True)

        with open(self.run_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Experiment directory: {self.run_dir}")
        if resume_from:
            print(f"Resuming from checkpoint: {resume_from}")

    def _generate_experiment_name(self) -> Path:
        """{model}_{timestamp}_{datasize}"""
        data_config = self.config['data']

        max_samples = self._run_cfg().get('max_samples', data_config.get('max_samples'))
        if max_samples:
            if max_samples >= 1_000_000:
                dataset_size = f"{max_samples // 1_000_000}M"
            elif max_samples >= 1_000:
                dataset_size = f"{max_samples // 1_000}k"
            else:
                dataset_size = str(max_samples)
        else:
            dataset_size = "full"

        storage_filter = data_config.get('storage_place_filter', '')
        if storage_filter:
            filter_clean = storage_filter.replace('%', '').replace('-', '')
            dataset_size = f"{filter_clean}_{dataset_size}"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.model_name}_{timestamp}_{dataset_size}"

        if self.config.get('experiment_suffix'):
            exp_name += f"_{self.config['experiment_suffix']}"

        return self.experiment_dir / self.model_name / exp_name

    def _get_experiment_dir_from_checkpoint(self, checkpoint_path: str) -> Path:
        path = Path(checkpoint_path)
        if path.suffix == '.ckpt':
            return path.parent.parent
        return path

    def _resolve_checkpoint_path(self, checkpoint_path: str) -> str:
        path = Path(checkpoint_path)

        if path.suffix == '.ckpt' and path.exists():
            return str(path)

        if path.is_dir():
            ckpt_dir = path / 'checkpoints' if (path / 'checkpoints').exists() else path
            checkpoints = list(ckpt_dir.glob('*.ckpt'))
            if checkpoints:
                best = min(
                    checkpoints,
                    key=lambda p: float(re.search(r'val_mpe=([\d.]+)', p.name).group(1))
                    if re.search(r'val_mpe=([\d.]+)', p.name) else float('inf')
                )
                return str(best)

        raise ValueError(f"Could not resolve checkpoint path: {checkpoint_path}")

    def create_data_module(self) -> WiFiFingerprintDataModule:
        data_config = self.config['data']
        target_scale = data_config.get('target_scale')
        # full data needs an explicit max_samples: null in the run
        if 'max_samples' not in self._run_cfg() and data_config.get('max_samples'):
            print(f"NOTE: run '{self.model_name}' inherits global "
                  f"data.max_samples={data_config['max_samples']} "
                  f"(set 'max_samples: null' in the run for full data)")
        return WiFiFingerprintDataModule(
            db_path=data_config['db_path'],
            train_split=data_config['train_split'],
            val_split=data_config['val_split'],
            test_split=data_config['test_split'],
            random_seed=self._run_cfg().get('random_seed', data_config['random_seed']),
            max_samples=self._run_cfg().get('max_samples', data_config.get('max_samples')),
            storage_place_filter=data_config.get('storage_place_filter'),
            deduplicate=self._run_cfg().get('deduplicate', data_config.get('deduplicate', False)),
            sampling=self._run_cfg().get('sampling', data_config.get('sampling', 'random')),
            reps_per_place=self._run_cfg().get('reps_per_place', data_config.get('reps_per_place', 10)),
            clean_fraction=self._run_cfg().get('clean_fraction', data_config.get('clean_fraction', 0.0)),
            clean_k=self._run_cfg().get('clean_k', data_config.get('clean_k', 9)),
            split_strategy=self._run_cfg().get('split_strategy', data_config.get('split_strategy', 'random')),
            max_aps=data_config['max_aps'],
            missing_value=data_config['missing_value'],
            ssid_filter=data_config.get('filter_ssid_prefix'),
            noise_coefficient=self._run_cfg().get(
                'noise_coefficient', self.config['training'].get('noise_coefficient', 0.0)),
            target_scale=tuple(target_scale) if target_scale else None,
            rssi_normalization=data_config.get('rssi_normalization', 'minmax'),
            rssi_min=data_config.get('rssi_min', -100.0),
            rssi_max=data_config.get('rssi_max', 0.0),
            ap_permutation_seed=self._run_cfg().get(
                'ap_permutation_seed', data_config.get('ap_permutation_seed')),
        )

    def _run_cfg(self) -> dict:
        return dict(self.config.get('runs', {}).get(self.model_name, {}))

    def create_model(self) -> pl.LightningModule:
        model_config = self._run_cfg()
        model_config.pop('model', None)
        training_config = self.config['training']
        data_config = self.config['data']

        kwargs = {
            'learning_rate': training_config['learning_rate'],
            'weight_decay': training_config['weight_decay'],
            'optimizer': training_config['optimizer'],
            'scheduler': model_config.pop('scheduler', training_config['scheduler']),
            'warmup_epochs': training_config.get('warmup_epochs', 5),
            'cosine_t_max': training_config.get('cosine_t_max'),
            'backbone_lr_mult': training_config.get('backbone_lr_mult', 1.0),
            'cdf_thresholds': self.config['metrics'].get('cdf_thresholds', [1.0, 2.0, 5.0, 10.0]),
            'target_scale': data_config.get('target_scale'),
        }
        kwargs.update(model_config)

        image_shape = kwargs.get('image_shape', data_config['image_shape'])
        if self.model_type in _IMAGE_MODELS:
            kwargs.setdefault('image_shape', image_shape)
            kwargs.setdefault('img_size', image_shape[0])
            kwargs.setdefault('in_chans', image_shape[2])
        else:
            bipartite_offset = 1 if kwargs.get('graph_topology') == 'bipartite' else 0
            kwargs.setdefault('num_aps', data_config['max_aps'] + bipartite_offset)

        kwargs.setdefault('dropout', training_config.get('dropout', 0.5))

        if self.model_type in ('transformer', 'transformer_cls'):
            kwargs.setdefault('missing_value', 0.0)

        cls = _import_model_class(self.model_type)
        params, has_var_kw = _accepted_params(cls)
        if not has_var_kw:
            kwargs = {k: v for k, v in kwargs.items() if k in params}

        model = cls(**kwargs)

        if (self.model_type in ('transformer', 'transformer_cls')
                and model_config.get('spatial_positional_encoding')):
            model.model.spatial_pos_encoder.load_positions(
                db_path=data_config['db_path'],
                sorted_aps=self._sorted_aps,
            )

        return model

    def create_dataloaders(self, data_module: WiFiFingerprintDataModule) -> tuple:
        data_config = self.config['data']
        db_path = data_config['db_path']

        m_cfg = self._run_cfg()

        if self.model_type in _IMAGE_MODELS:
            kind = 'image'
        elif self.model_type in _GRAPH_MODELS:
            kind = 'graph'
        else:
            kind = None
        transform = build_transform(
            kind, m_cfg, data_config,
            db_path=db_path, sorted_aps=data_module.get_sorted_aps(),
        )

        data_module.dataset_kwargs['transform'] = transform

        if self.model_type in _GRAPH_MODELS:
            try:
                from torch_geometric.loader import DataLoader as PyGDataLoader
                Loader = PyGDataLoader
            except ImportError:
                print("Warning: torch_geometric not found, using standard DataLoader")
                Loader = DataLoader
        else:
            Loader = DataLoader

        loader_kwargs = dict(
            batch_size=m_cfg.get('batch_size', data_config['batch_size']),
            num_workers=m_cfg.get('num_workers', data_config['num_workers']),
            pin_memory=True,
        )
        if (self.model_type in ('transformer', 'transformer_cls')
                and m_cfg.get('variable_length')):
            loader_kwargs['collate_fn'] = sparse_transformer_collate_fn

        train_kwargs = dict(loader_kwargs, drop_last=True)

        return (
            Loader(data_module.get_dataset('train'), shuffle=True, **train_kwargs),
            Loader(data_module.get_dataset('val'), shuffle=False, **loader_kwargs),
            Loader(data_module.get_dataset('test'), shuffle=False, **loader_kwargs),
        )

    def create_callbacks(self) -> list:
        tc = self.config['training']
        return [
            ModelCheckpoint(
                dirpath=self.run_dir / "checkpoints",
                filename='{epoch}-{val_mpe:.2f}',
                monitor=tc.get('checkpoint_metric', 'val_mpe'),
                mode=tc.get('checkpoint_mode', 'min'),
                save_top_k=tc.get('save_top_k', 3),
                verbose=True,
            ),
            EarlyStopping(
                monitor='val_mpe',
                patience=tc.get('early_stopping_patience', 10),
                mode='min',
                verbose=True,
            ),
            LearningRateMonitor(logging_interval='epoch'),
        ]

    def create_logger(self) -> TensorBoardLogger:
        return TensorBoardLogger(save_dir=str(self.run_dir), name="tensorboard", version="")

    def train(self) -> pl.LightningModule:
        print(f"\n{'='*60}\nStarting training: {self.model_name}\n{'='*60}\n")

        data_module = self.create_data_module()
        self._sorted_aps = data_module.get_sorted_aps()
        model = self.create_model()
        train_loader, val_loader, test_loader = self.create_dataloaders(data_module)

        # cls models use the train-split centroids as classes
        if hasattr(model, 'set_centroids'):
            import numpy as np
            centroids = np.unique(train_loader.dataset._targets, axis=0)
            model.set_centroids(torch.from_numpy(centroids))

        with open(self.run_dir / "sorted_aps.json", 'w') as f:
            json.dump(data_module.get_sorted_aps(), f, indent=2)

        trainer = pl.Trainer(
            max_epochs=self._run_cfg().get('max_epochs', self.config['training']['max_epochs']),
            accelerator=self.config['hardware']['accelerator'],
            devices=self.config['hardware']['devices'],
            precision=self.config['hardware']['precision'],
            deterministic=self.config['hardware']['deterministic'],
            callbacks=self.create_callbacks(),
            logger=self.create_logger(),
            gradient_clip_val=self.config['training'].get('gradient_clip_val'),
            log_every_n_steps=self.config['logging'].get('log_every_n_steps', 50),
        )

        if hasattr(model, 'pretrain_encoder'):
            if torch.cuda.is_available():
                model = model.cuda()
            model.pretrain_encoder(train_loader)
            model = model.cpu()

        ckpt_path = None
        if self.resume_from:
            ckpt_path = self._resolve_checkpoint_path(self.resume_from)
            print(f"Resuming training from: {ckpt_path}")

        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

        print(f"\n{'='*60}\nRunning final evaluation on test set...\n{'='*60}\n")
        trainer.test(model, test_loader, ckpt_path='best')

        records = getattr(model, '_test_records', None)
        if self.config.get('logging', {}).get('save_predictions') and records and records["id"]:
            import numpy as np
            out = self.run_dir / "test_predictions.npz"
            np.savez(out,
                     id=np.asarray(records["id"], dtype=np.int64),
                     pred=np.asarray(records["pred"], dtype=np.float32),
                     true=np.asarray(records["true"], dtype=np.float32))
            print(f"Saved {len(records['id'])} test predictions (metres) to {out}")

        trainer.save_checkpoint(self.run_dir / "final_model.ckpt")
        print(f"\nTraining complete! Results saved to: {self.run_dir}")

        return model
