"""Feature extraction: MERCI motif scores, sequence descriptors, protein language model embeddings."""
import contextlib
import importlib
import io
import logging
import os
import shutil
import ssl
import subprocess
import sys
import tempfile

import numpy as np
import torch

log = logging.getLogger(__name__)

ESM2_MODELS = {
    'ESM2-t6': 'esm2_t6_8M_UR50D',
    'ESM2-t12': 'esm2_t12_35M_UR50D',
    'ESM2-t30': 'esm2_t30_150M_UR50D',
    'ESM2-t33': 'esm2_t33_650M_UR50D',
    'ESM2-t36': 'esm2_t36_3B_UR50D',
}

CONVENTIONAL_DESCRIPTORS = [
    'AAC', 'APAAC', 'ASDC', 'CKSAAGP-type-1', 'CKSAAGP-type-2', 'CKSAAP-type-1', 'CKSAAP-type-2',
    'CTDC', 'CTDD', 'CTDT', 'CTriad', 'DDE', 'DPC-type-1', 'DPC-type-2', 'GDPC-type-2',
    'DistancePair', 'PAAC', 'QSOrder',
]

# feature name -> (bio_embeddings class, optional weight file of the ESM-1 family)
EMBEDDERS = {
    'Seq2Vec': ('SeqVecEmbedder', None),
    'Bepler': ('BeplerEmbedder', None),
    'CPCProt': ('CPCProtEmbedder', None),
    'ESM1b': ('ESM1bEmbedder', 'esm1b_t33_650M_UR50S.pt'),
    'ESM1v': ('ESM1vEmbedder', 'esm1v_t33_650M_UR90S_1.pt'),
    'ESM': ('ESMEmbedder', 'esm1_t34_670M_UR50S.pt'),
    'FastText': ('FastTextEmbedder', None),
    'GloVe': ('GloveEmbedder', None),
    'PRNN': ('PLUSRNNEmbedder', None),
    'PTAB': ('ProtTransAlbertBFDEmbedder', None),
    'PTBB': ('ProtTransBertBFDEmbedder', None),
    'PTB': ('ProtTransT5BFDEmbedder', None),
    'PTU': ('ProtTransT5UniRef50Embedder', None),
    'PTXU': ('ProtTransT5XLU50Embedder', None),
    'PTXLU': ('ProtTransXLNetUniRef100Embedder', None),
    'Word2Vec': ('Word2VecEmbedder', None),
}
CPU_ONLY_EMBEDDERS = {'FastText', 'GloVe', 'Word2Vec'}

# motif library file, MERCI alphabet, score contribution
MERCI_LIBRARIES = [
    ('pos_motif_1.txt', 'NONE', 0.5),
    ('pos_motif_2.txt', 'KOOLMAN-ROHM', 0.5),
    ('neg_motif_1.txt', 'NONE', -0.5),
    ('neg_motif_2.txt', 'KOOLMAN-ROHM', -0.5),
]


def check_environment(esm_dir):
    """Raise a RuntimeError listing every missing or unusable requirement."""
    problems = []
    if shutil.which('perl') is None:
        problems.append('perl not found in PATH (required for MERCI)')
    if not os.path.isdir(os.path.join(esm_dir, 'esm')):
        problems.append(f"ESM repository not found in '{esm_dir}' "
                        f"(git clone https://github.com/facebookresearch/esm.git {esm_dir})")
    for module in ('bio_embeddings.embed', 'iFeatureOmegaCLI', 'peptidy'):
        try:
            importlib.import_module(module)
        except Exception as e:
            problems.append(f'cannot import {module}: {type(e).__name__}: {e}')
    try:
        import bio_embeddings.embed as be
        missing = sorted(cls for cls, _ in EMBEDDERS.values() if not hasattr(be, cls))
        if missing:
            problems.append(f'bio_embeddings lacks {", ".join(missing)} (install fair-esm==0.4.0)')
    except Exception:
        pass
    if problems:
        raise RuntimeError('Unusable environment:\n  - ' + '\n  - '.join(problems) +
                           '\nSee "Installing environment" in the README.')


def write_fasta(ids, seqs, path):
    with open(path, 'w') as f:
        for i, s in zip(ids, seqs):
            f.write(f'>{i}\n{s}\n')


def _merci_hits(path):
    hits, in_coverage = set(), False
    if not os.path.exists(path):
        return hits
    with open(path) as f:
        for line in f:
            if 'COVERAGE' in line:
                in_coverage = True
            elif in_coverage and '(' in line:
                hits.add(line.split('(')[0].strip().replace('>', ''))
    return hits


def extract_merci_scores(ids, seqs, locator, motif_dir):
    """MERCI score per sequence: +0.5 per positive library hit, -0.5 per negative library hit."""
    if shutil.which('perl') is None:
        raise RuntimeError('perl is required for MERCI but was not found in PATH')
    tmp = tempfile.mkdtemp(prefix='hlppred2_merci_')
    try:
        fasta = os.path.join(tmp, 'input.fasta')
        write_fasta(ids, seqs, fasta)
        scores = dict.fromkeys(ids, 0.0)
        for library, alphabet, weight in MERCI_LIBRARIES:
            out = os.path.join(tmp, library + '.out')
            subprocess.run(['perl', locator, '-p', fasta, '-i', os.path.join(motif_dir, library),
                            '-o', out, '-c', alphabet],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=tmp)
            for i in _merci_hits(out) & scores.keys():
                scores[i] += weight
        return scores
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _import_esm2(esm_dir):
    """Import ESM-2 from a local clone; bio_embeddings pins an older `esm` without ESM-2."""
    for k in [k for k in sys.modules if k == 'esm' or k.startswith('esm.')]:
        del sys.modules[k]
    saved_path = list(sys.path)
    sys.path = [esm_dir] + [p for p in sys.path if 'site-packages' not in p]
    try:
        importlib.invalidate_caches()
        from esm import pretrained
    finally:
        sys.path = saved_path
    return pretrained


def _extract_esm2(ids, seqs, names, esm_dir, device, features):
    ssl._create_default_https_context = ssl._create_unverified_context
    pretrained = _import_esm2(esm_dir)
    try:
        for feat in names:
            model = None
            try:
                log.info('%s (%s)', feat, ESM2_MODELS[feat])
                model, alphabet = pretrained.load_model_and_alphabet(ESM2_MODELS[feat])
                model.eval()
                run_device = 'cpu'
                if device == 'cuda':
                    try:
                        model, run_device = model.cuda(), 'cuda'
                    except RuntimeError as e:
                        if 'out of memory' not in str(e):
                            raise
                        log.warning('%s: GPU out of memory, using CPU', feat)
                        torch.cuda.empty_cache()
                        model = model.cpu()
                convert = alphabet.get_batch_converter()
                features[feat] = {}
                for i, s in zip(ids, seqs):
                    _, _, tokens = convert([(i, s)])
                    with torch.no_grad():
                        rep = model(tokens.to(run_device), repr_layers=[model.num_layers])
                    emb = rep['representations'][model.num_layers][0, 1:len(s) + 1]
                    features[feat][i] = emb.mean(0).cpu().numpy().astype(np.float32)
            except Exception as e:
                log.warning('%s: extraction failed: %s', feat, e)
            finally:
                del model
                if device == 'cuda':
                    torch.cuda.empty_cache()
    finally:
        for k in [k for k in sys.modules if k == 'esm' or k.startswith('esm.')]:
            del sys.modules[k]
        if esm_dir in sys.path:
            sys.path.remove(esm_dir)


def _extract_conventional(ids, seqs, names, features):
    import iFeatureOmegaCLI
    tmp = tempfile.mkdtemp(prefix='hlppred2_ifeature_')
    try:
        fasta = os.path.join(tmp, 'input.fasta')
        write_fasta(ids, seqs, fasta)
        protein = iFeatureOmegaCLI.iProtein(fasta)
        params = os.path.join(os.path.dirname(iFeatureOmegaCLI.__file__), 'parameters',
                              'Protein_parameters_setting.json')
        if os.path.exists(params):
            with contextlib.redirect_stdout(io.StringIO()):
                protein.import_parameters(params)
        for desc in names:
            try:
                log.info(desc)
                protein.get_descriptor(desc.replace('-', ' '))
                features[desc] = {idx.split()[0]: row.values.astype(np.float32)
                                  for idx, row in protein.encodings.iterrows()}
            except Exception as e:
                log.warning('%s: extraction failed: %s', desc, e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _extract_pde(ids, seqs, features):
    import peptidy
    features['PDE'] = {
        i: np.array(peptidy.encoding.peptide_descriptor_encoding(s)).flatten().astype(np.float32)
        for i, s in zip(ids, seqs)
    }


def _load_embedder(feat, device, weights_dir):
    import bio_embeddings.embed as be
    cls_name, weight_file = EMBEDDERS[feat]
    kwargs = {}
    if weight_file and weights_dir and os.path.exists(os.path.join(weights_dir, weight_file)):
        kwargs['model_file'] = os.path.join(weights_dir, weight_file)
    if feat == 'ESM1v':
        kwargs['ensemble_id'] = 1
    if device and feat not in CPU_ONLY_EMBEDDERS:
        kwargs['device'] = device
    return getattr(be, cls_name)(**kwargs)


def _seqvec_mean(emb):
    """SeqVec returns [3 layers, L, 1024]; the models use the mean over layers."""
    emb = np.asarray(emb)
    return torch.tensor(emb.mean(axis=0) if emb.ndim == 3 else emb, dtype=torch.float32)


def _extract_embedders(ids, seqs, names, device, weights_dir, features, seqvec):
    cpu = 'cpu' if device == 'cpu' else None
    for feat in names:
        embedder = None
        try:
            log.info(feat)
            try:
                embedder = _load_embedder(feat, cpu, weights_dir)
            except RuntimeError as e:
                if 'out of memory' not in str(e):
                    raise
                log.warning('%s: GPU out of memory, using CPU', feat)
                torch.cuda.empty_cache()
                embedder = _load_embedder(feat, 'cpu', weights_dir)
            features[feat] = {}
            for i, s in zip(ids, seqs):
                emb = embedder.embed(s)
                features[feat][i] = embedder.reduce_per_protein(emb).astype(np.float32)
                if feat == 'Seq2Vec' and seqvec is not None:
                    seqvec[i] = _seqvec_mean(emb)
        except Exception as e:
            log.warning('%s: extraction failed: %s', feat, e)
        finally:
            del embedder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def required_feature_types(base_model_names):
    """Base model names have the form '<ALGORITHM>_<FEATURE>', e.g. 'XGB_ESM2-t6'."""
    return sorted({n.split('_', 1)[1] for n in base_model_names})


def extract_raw_features(ids, seqs, feature_types, esm_dir, weights_dir=None, device='cpu', seqvec=None):
    """Return {feature type: {id: vector}}. Per-residue SeqVec embeddings are collected in `seqvec`."""
    wanted = set(feature_types)
    unknown = wanted - set(ESM2_MODELS) - set(CONVENTIONAL_DESCRIPTORS) - set(EMBEDDERS) - {'PDE'}
    if unknown:
        raise ValueError(f'Unknown feature types: {sorted(unknown)}')
    features = {}
    steps = (
        ('ESM-2', [f for f in ESM2_MODELS if f in wanted],
         lambda n: _extract_esm2(ids, seqs, n, esm_dir, device, features)),
        ('Sequence descriptors', [f for f in CONVENTIONAL_DESCRIPTORS if f in wanted],
         lambda n: _extract_conventional(ids, seqs, n, features)),
        ('PDE', [f for f in ('PDE',) if f in wanted], lambda n: _extract_pde(ids, seqs, features)),
        ('Protein language models', [f for f in EMBEDDERS if f in wanted],
         lambda n: _extract_embedders(ids, seqs, n, device, weights_dir, features, seqvec)),
    )
    for title, names, extract in steps:
        if names:
            log.info('Extracting features: %s', title)
            extract(names)
    return features


def extract_seqvec_residue(ids, seqs, device='cpu', done=None):
    """Per-residue SeqVec embeddings {id: tensor[L, 1024]}; sequences in `done` are skipped."""
    out = dict(done or {})
    todo = [(i, s) for i, s in zip(ids, seqs) if i not in out]
    if todo:
        log.info('Extracting per-residue SeqVec embeddings')
        embedder = _load_embedder('Seq2Vec', 'cpu' if device == 'cpu' else None, None)
        for i, s in todo:
            out[i] = _seqvec_mean(embedder.embed(s))
        del embedder
    return out
