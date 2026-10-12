"""Bind a TDD exported checkpoint to its recorded nnU-Net model snapshot."""
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_binding(checkpoint, training_run):
    """Reject missing, inconsistent or changed artifacts before model execution."""
    checkpoint = Path(checkpoint).resolve()
    training_run = Path(training_run).resolve()
    summary = training_run / 'report/summary.json'
    best = json.loads(summary.read_text())['best_experiment']
    config = best['config']
    recorded = Path(best['best_model_path']).resolve()
    recorded.relative_to(training_run)
    fold = str(config['fold'])
    if recorded.parent.name != 'fold_' + fold:
        raise ValueError('Recorded fold and checkpoint folder disagree')
    model_folder = recorded.parent.parent
    if model_folder.name.split('__')[0] != config['nnunet_trainer'] or model_folder.name.split('__')[-1] != config['configuration']:
        raise ValueError('Recorded trainer/configuration and model folder disagree')
    for name in ('plans.json', 'dataset.json'):
        if not (model_folder / name).is_file():
            raise ValueError('Missing nnU-Net model metadata: ' + name)
    digest = sha256_file(checkpoint)
    if sha256_file(recorded) != digest:
        raise ValueError('Exported checkpoint differs from the recorded model snapshot')
    result = {
        'model_folder': str(model_folder), 'fold': fold,
        'checkpoint_name': recorded.name, 'checkpoint_sha256': digest,
        'configuration': str(config['configuration']),
        'trainer': str(config['nnunet_trainer']),
        'recorded_checkpoint': str(recorded),
        'selection_summary': str(summary), 'selection_summary_sha256': sha256_file(summary),
        'plans_sha256': sha256_file(model_folder / 'plans.json'),
        'dataset_json_sha256': sha256_file(model_folder / 'dataset.json'),
    }
    trainer_source = config.get('trainer_definition_path')
    if trainer_source:
        result['trainer_source'] = str(Path(trainer_source).resolve())
        result['trainer_source_sha256'] = sha256_file(trainer_source)
    return result


def verify_model_binding(binding, checkpoint):
    folder = Path(binding['model_folder'])
    recorded = folder / ('fold_' + binding['fold']) / binding['checkpoint_name']
    for path, expected in (
        (checkpoint, binding['checkpoint_sha256']),
        (recorded, binding['checkpoint_sha256']),
        (folder / 'plans.json', binding['plans_sha256']),
        (folder / 'dataset.json', binding['dataset_json_sha256']),
        (binding['selection_summary'], binding['selection_summary_sha256']),
    ):
        if sha256_file(path) != expected:
            raise ValueError('Bound model artifact changed: ' + str(path))
    if binding.get('trainer_source') and sha256_file(binding['trainer_source']) != binding['trainer_source_sha256']:
        raise ValueError('Bound trainer source changed')
