"""Reproducible SMCF-Net training with fixed split CSVs and validation-only selection."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, platform, time, random
import importlib.metadata
from pathlib import Path
from typing import Any
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from benchmark.data import ManifestDataset, PairedTransform, resolve_data_path
from benchmark.history import write_history
from benchmark.losses import BCEDiceLoss, TverskyLoss
from benchmark.metrics import BinaryConfusionMatrix
from models.smcf_net import SMCFNet, SMCFNetModelConfig, VARIANTS
from models.registry import flood_probabilities
ROOT=Path(__file__).resolve().parents[1]
DATASETS=('s1gfloods','ombrias1')
def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

def make_loader(
    root: Path,
    manifest: Path,
    data_config: dict[str, Any],
    train: bool,
    seed: int,
) -> DataLoader:
    transform = PairedTransform(
        size=int(data_config.get("image_size", 256)),
        train=train,
        random_exchange=train and bool(data_config.get("random_exchange", False)),

    )
    dataset = ManifestDataset(root, manifest, transform)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(data_config.get("batch_size", 16)),
        shuffle=train,
        num_workers=int(data_config.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        generator=generator,
        persistent_workers=int(data_config.get("num_workers", 4)) > 0,
    )

def build_criterion(config: dict[str, Any]) -> torch.nn.Module:
    name = str(config["name"])
    if name == "tversky":
        return TverskyLoss(
            alpha=float(config.get("alpha", 0.3)),
            beta=float(config.get("beta", 0.7)),
            smooth=float(config.get("smooth", 1.0)),
        )
    if name == "bce_dice":
        return BCEDiceLoss()
    raise ValueError(f"unsupported loss {name!r}")

def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    step: int,
    max_steps: int,
    warmup_steps: int,
) -> float:
    if step < warmup_steps:
        lr = base_lr * (0.1 + 0.9 * (step + 1) / warmup_steps)
    else:
        progress = min(step / max_steps, 1.0)
        lr = base_lr * (1.0 - progress) ** 0.9
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

def safe_output(path):
    path = Path(path).resolve()
    for folder in ('models', 'benchmark', 'data', 'configs', 'splits', 'private', '.git'):
        protected = ROOT / folder
        if path == protected or protected in path.parents or path in protected.parents:
            raise ValueError(f'Output would overwrite protected repository content: {path}')
    return path

def manifest(dataset, split):
    return ROOT / 'splits' / dataset / f'{split}.csv'

def loader(config, split, train=False):
    return make_loader(ROOT, manifest(config['dataset'], split), config['data'], train, config['seed'])

def objective(config, predictions, batch, device):
    criterion = build_criterion(config['loss'])
    from benchmark.losses import deep_supervision_loss
    return deep_supervision_loss(predictions, batch['mask'].to(device), criterion, batch['valid'].to(device), 'mean')

@torch.no_grad()
def evaluate(model, config, split, device, thresholds, perturbation=None):
    if split != 'val':
        raise ValueError('Public trainer evaluates validation only')
    model.eval()
    thresholds = np.asarray(thresholds, dtype=np.float64)
    positive = np.zeros(len(thresholds) + 1, dtype=np.int64)
    negative = positive.copy()
    per_image = []
    for batch in loader(config, split):
        pre, post = (batch['pre'].to(device), batch['post'].to(device))
        if perturbation == 'swap':
            pre, post = (post, pre)
        elif perturbation == 'brightness':
            post = post + 0.2
        elif perturbation == 'offset':
            post = torch.nn.functional.pad(post, (2, 0, 0, 0), mode='replicate')[:, :, :, :-2]
        probability = flood_probabilities(config['model_name'], model(pre, post))[0].cpu().numpy()
        if not np.isfinite(probability).all():
            raise ValueError('Non-finite predictions')
        truth = batch['mask'].numpy() >= 0.5
        valid = batch['valid'].numpy().astype(bool)
        bins = np.searchsorted(thresholds, probability[valid], side='right')
        positive += np.bincount(bins[truth[valid]], minlength=len(positive))
        negative += np.bincount(bins[~truth[valid]], minlength=len(negative))
    tp = np.cumsum(positive[::-1])[::-1][1:]
    fp = np.cumsum(negative[::-1])[::-1][1:]
    results = [dict(threshold=float(t), **BinaryConfusionMatrix(int(a), int(b), int(negative.sum() - b), int(positive.sum() - a)).compute()) for t, a, b in zip(thresholds, tp, fp)]
    return (results, per_image)

def verify(config):
    counts = {}
    for dataset in (config['dataset'],):
        counts[dataset] = {}
        identities = []
        for split in ('train', 'val', 'test'):
            rows = list(csv.DictReader(manifest(dataset, split).open(newline='', encoding='utf-8')))
            ids = {r['id'] for r in rows}
            assert len(ids) == len(rows), 'Duplicate sample ID'
            assert all((not ids & previous for previous in identities)), 'Overlapping split IDs'
            identities.append(ids)
            for row in rows:
                for key in ('pre', 'post', 'mask'):
                    if not resolve_data_path(ROOT, row[key]).is_file():
                        raise FileNotFoundError(resolve_data_path(ROOT, row[key]))
            counts[dataset][split] = len(rows)
        size = int(config['data'].get('image_size', 256))
        sample = ManifestDataset(ROOT, manifest(dataset, 'val'), PairedTransform(size=size, train=False))[0]
        assert sample['pre'].shape == (3, size, size)
    print(json.dumps(dict(counts=counts, note='ID disjointness does not prove event/geographic independence'), indent=2))

def train(config, output, device):
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(config['seed'], True)
    write(output / 'config.json', config)
    write(output / 'provenance.json', provenance(config))
    model = construct(config, pretrained=config['model'].get('pretrained', True)).to(device)
    training = config['training']
    optimizer = torch.optim.Adam(model.parameters(), lr=training['learning_rate'], betas=tuple(training['betas']), eps=training['epsilon'], weight_decay=training['weight_decay'])
    train_loader = loader(config, 'train', True)
    if not len(train_loader):
        raise ValueError('No complete training batches')
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    history, step, best = ([], 0, -1.0)
    for epoch in range(math.ceil(training['max_steps'] / len(train_loader))):
        model.train()
        losses = []
        confusion = BinaryConfusionMatrix()
        for batch in train_loader:
            if step >= training['max_steps']:
                break
            lr = adjust_learning_rate(optimizer, training['learning_rate'], step, training['max_steps'], training['warmup_steps'])
            predictions = flood_probabilities(config['model_name'], model(batch['pre'].to(device), batch['post'].to(device)))
            loss = objective(config, predictions, batch, device)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            confusion.update(predictions[0].detach(), batch['mask'].to(device), valid_mask=batch['valid'].to(device))
            step += 1
        validation = evaluate(model, config, 'val', device, [0.5])[0][0]
        record = dict(epoch=epoch + 1, step=step, lr=lr, train_loss=float(np.mean(losses)), val_loss=None, train=confusion.compute(), val={k: v for k, v in validation.items() if k != 'threshold'})
        history.append(record)
        write_history(output, history)
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=config, epoch=epoch + 1, step=step, validation_f1=validation['f1'])
        torch.save(checkpoint, output / 'last.pt')
        if validation['f1'] >= best:
            best = validation['f1']
            torch.save(checkpoint, output / 'best.pt')
        print(json.dumps(dict(epoch=epoch + 1, step=step, validation_f1=validation['f1'])), flush=True)
    if device.type == 'cuda': torch.cuda.synchronize(device)
    write(output / 'training_complete.json', dict(final_step=step,
        parameters=sum(p.numel() for p in model.parameters()),
        training_wall_seconds=time.perf_counter()-training_started,
        training_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        timing_includes=['data_loading','training','epoch_validation','checkpoint_saving'],
        timing_excludes=['model_initialization','pretrained_download'],
        test_evaluated=False, exploratory=step != 40000))


def provenance(config):
    sources=list((ROOT/'benchmark').glob('*.py'))+list((ROOT/'models').glob('*.py'))
    return dict(manifests={s:sha(manifest(config['dataset'],s)) for s in ('train','val','test')},
                sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
                python=platform.python_version(),torch=torch.__version__,platform=platform.platform(),
                packages={p:importlib.metadata.version(p) for p in ('numpy','timm','PyYAML','torch','torchvision')},
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

def construct(config,pretrained=False):
    return SMCFNet(SMCFNetModelConfig(**dict(config['model'],pretrained=pretrained)),config['variant'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/study.yaml')
    p.add_argument('--dataset',choices=DATASETS,default='s1gfloods')
    p.add_argument('--seed',type=int,default=23534)
    p.add_argument('--variant',choices=VARIANTS,default='parallel')
    p.add_argument('--output',type=Path)
    p.add_argument('--device',default='cuda')
    p.add_argument('--data-root',type=Path,help='External data/ directory containing datasets/')
    args=p.parse_args()
    if args.data_root:
        os.environ['SMCF_NET_DATA_ROOT'] = str(args.data_root.resolve())
    config=yaml.safe_load(args.config.read_text())
    config.update(dataset=args.dataset,seed=args.seed,variant=args.variant,model_name='smcf_net')
    if args.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    verify(config)
    train(config,safe_output(args.output or ROOT/'results/study'/args.dataset/f'smcf_net-{args.variant}-seed{args.seed}'),torch.device(args.device))

if __name__=='__main__': main()
