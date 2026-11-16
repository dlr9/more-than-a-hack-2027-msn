"""Bounded CPU adapter to one pinned upstream checkpoint, never a model clone."""
import hashlib
import json
import pathlib
import shutil
import sys

ORIGINAL_COMMIT = '1b7c6f85a6b78c92e4543fde72ca2b86a933f997'
WEIGHT_SHA256 = 'c82235a17416354b5544f85721ca850304765737a2d9d5435dcdefd52a0d3700'


def load_original(root):
    """Strict CPU weights-only load. Missing runtime raises, never fabricated output.

    Core SetLSTM is tensor-only. models.utils imports PyG solely for an unused
    graph collator; the local CPU source copy drops that import and postpones
    its annotations. No neural operation, parameter or CRF algorithm changes.
    Original files remain intact; copy hashes and modification are recorded.
    """
    root = pathlib.Path(root)
    weight = root/'original140.pt'
    if hashlib.sha256(weight.read_bytes()).hexdigest() != WEIGHT_SHA256:
        raise ValueError('checkpoint hash mismatch')
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    manifest = json.loads((root/'upstream_manifest.json').read_text())
    if manifest['commit'] != ORIGINAL_COMMIT: raise ValueError('source version mismatch')
    source = root/'upstream'; cpu_source = root/'cpu_source'
    for row in manifest['files']:
        path = source/row['path']; data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('source digest mismatch')
        if row['path'].startswith(('models/', 'set_transformer/', 'datatools/')):
            target = cpu_source/row['path']; target.parent.mkdir(parents=True, exist_ok=True)
            if row['path'] == 'models/utils.py':
                data = b'from __future__ import annotations\n' + data.replace(
                    b'from torch_geometric.data import Batch, Data\n', b'')
            if target.exists() and target.read_bytes() != data:
                raise ValueError('CPU source copy differs')
            if not target.exists(): target.write_bytes(data)
    sys.path.insert(0, str(cpu_source.resolve()))
    from models.utils import build_model
    args = json.loads((source/'saved/140/args.json').read_text())
    model = build_model(args, torch.device('cpu'))
    state = torch.load(weight, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True); model.eval()
    return model, torch


def forward(model, torch, window):
    x = torch.tensor(window['nodes'].tolist(), dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        macro, logits, embedding = model(x)
        keep = model.valid_orig_edge_ids
        emissions = logits[:, :, keep]; emb = embedding[:, :, keep]
        path = model.crf.decode(emissions, emb)
        src = model.comp_src[path]; dst = model.comp_dst[path]
        # Emission marginal is not a CRF posterior; explicitly named as such.
        probabilities = emissions.softmax(-1)
        left = probabilities[:, :, model.comp_src < 11].sum(-1)
        right = probabilities[:, :, (model.comp_src >= 11) & (model.comp_src < 22)].sum(-1)
        outside = 1-left-right
        p_home = left if window['left_team'] == 0 else right
    return dict(path_src=src[0].tolist(), path_dst=dst[0].tolist(),
                p_home_emission=p_home[0].tolist(), p_outside_emission=outside[0].tolist(),
                probability_semantics='emission_marginal_not_calibrated_CRF_posterior',
                n_parameters=sum(p.numel() for p in model.parameters()),
                finite=bool(torch.isfinite(logits).all()))
