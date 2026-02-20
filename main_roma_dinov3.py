"""
Train SALAD with RoMaV2 DINOv3 backbone and configurable finetuning depth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train SALAD with RoMaV2 DINOv3 backbone and optional block finetuning.'
    )

    # Required paths
    parser.add_argument(
        '--romav2-ckpt-path',
        type=Path,
        required=True,
        help="Path to RoMaV2 checkpoint containing descriptor keys prefixed by 'f.'.",
    )
    parser.add_argument(
        '--gsv-root',
        type=Path,
        default=Path('../data/GSVCities'),
        help='Path to GSV-Cities root containing Dataframes/ and Images/.',
    )

    # Data
    parser.add_argument('--batch-size', type=int, default=60)
    parser.add_argument('--img-per-place', type=int, default=4)
    parser.add_argument('--min-img-per-place', type=int, default=4)
    parser.add_argument('--shuffle-all', action='store_true')
    parser.add_argument('--random-sample-from-each-place', action='store_true', default=True)
    parser.add_argument('--no-random-sample-from-each-place', action='store_false', dest='random_sample_from_each_place')
    parser.add_argument('--image-size', nargs=2, type=int, default=(224, 224))
    parser.add_argument('--num-workers', type=int, default=10)
    parser.add_argument('--no-data-stats', action='store_true')
    parser.add_argument(
        '--val-set-names',
        nargs='+',
        default=['pitts30k_val', 'pitts30k_test'],
        help='Validation sets. Supported by current datamodule: pitts30k_val, pitts30k_test, msls_val.',
    )

    # Backbone / aggregator
    parser.add_argument('--disable-norm-layer', action='store_true')
    parser.add_argument(
        '--num-trainable-blocks',
        type=int,
        default=0,
        help='Number of last DINOv3 transformer blocks to finetune. 0 keeps full backbone frozen.',
    )
    parser.add_argument('--num-clusters', type=int, default=64)
    parser.add_argument('--cluster-dim', type=int, default=128)
    parser.add_argument('--token-dim', type=int, default=256)

    # Optimization
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--weight-decay', type=float, default=9.5e-9)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr-sched', type=str, default='linear')
    parser.add_argument('--lr-start-factor', type=float, default=1.0)
    parser.add_argument('--lr-end-factor', type=float, default=0.2)
    parser.add_argument('--lr-total-iters', type=int, default=4000)
    parser.add_argument('--loss-name', type=str, default='MultiSimilarityLoss')
    parser.add_argument('--miner-name', type=str, default='MultiSimilarityMiner')
    parser.add_argument('--miner-margin', type=float, default=0.1)
    parser.add_argument('--faiss-gpu', action='store_true')

    # Trainer
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--num-nodes', type=int, default=1)
    parser.add_argument('--default-root-dir', type=str, default='./logs/')
    parser.add_argument('--num-sanity-val-steps', type=int, default=0)
    parser.add_argument('--precision', type=str, default='16-mixed')
    parser.add_argument('--max-epochs', type=int, default=4)
    parser.add_argument('--check-val-every-n-epoch', type=int, default=1)
    parser.add_argument('--save-top-k', type=int, default=3)
    parser.add_argument('--log-every-n-steps', type=int, default=20)

    return parser


def _validate_paths(args: argparse.Namespace) -> None:
    if not args.romav2_ckpt_path.is_file():
        raise FileNotFoundError(
            f"RoMaV2 checkpoint file does not exist: {args.romav2_ckpt_path}"
        )
    if not args.gsv_root.exists():
        raise FileNotFoundError(
            f"GSV root path does not exist: {args.gsv_root}"
        )
    if args.num_trainable_blocks < 0:
        raise ValueError(
            f"num_trainable_blocks must be >= 0, got {args.num_trainable_blocks}."
        )


def main() -> None:
    import pytorch_lightning as pl
    from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
    from vpr_model import VPRModel

    parser = _build_parser()
    args = parser.parse_args()
    _validate_paths(args)

    datamodule = GSVCitiesDataModule(
        batch_size=args.batch_size,
        img_per_place=args.img_per_place,
        min_img_per_place=args.min_img_per_place,
        shuffle_all=args.shuffle_all,
        random_sample_from_each_place=args.random_sample_from_each_place,
        image_size=tuple(args.image_size),
        num_workers=args.num_workers,
        show_data_stats=not args.no_data_stats,
        val_set_names=args.val_set_names,
        base_path=str(args.gsv_root),
    )

    model = VPRModel(
        backbone_arch='romav2_dinov3_vitl16',
        backbone_config={
            'romav2_ckpt_path': str(args.romav2_ckpt_path),
            'norm_layer': not args.disable_norm_layer,
            'return_token': True,
            'num_trainable_blocks': args.num_trainable_blocks,
        },
        agg_arch='SALAD',
        agg_config={
            'num_channels': 1024,
            'num_clusters': args.num_clusters,
            'cluster_dim': args.cluster_dim,
            'token_dim': args.token_dim,
        },
        lr=args.lr,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        lr_sched=args.lr_sched,
        lr_sched_args={
            'start_factor': args.lr_start_factor,
            'end_factor': args.lr_end_factor,
            'total_iters': args.lr_total_iters,
        },
        loss_name=args.loss_name,
        miner_name=args.miner_name,
        miner_margin=args.miner_margin,
        faiss_gpu=args.faiss_gpu,
    )

    monitor_name = f'{args.val_set_names[0]}/R1'
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor=monitor_name,
        filename=f'{model.encoder_arch}' + '_({epoch:02d})_R1[{' + monitor_name + ':.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=args.save_top_k,
        save_last=True,
        mode='max'
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir=args.default_root_dir,
        num_nodes=args.num_nodes,
        num_sanity_val_steps=args.num_sanity_val_steps,
        precision=args.precision,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=[checkpoint_cb],
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=args.log_every_n_steps,
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    main()
