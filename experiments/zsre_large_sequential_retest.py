import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

import argparse
import json
import math
from copy import deepcopy
from time import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    MQUAKEDataset,
    get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_mquake import compute_rewrite_quality_mquake
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from AlphaEdit import AlphaEditHyperParams
from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov
from PMPh import PMPhHyperParams
from PMPh.PMPh_main import apply_PMPh_to_model
from memit import MEMITHyperParams
from memit.memit_main import apply_memit_to_model
from memit.memit_rect_main import apply_memit_rect_to_model
from memit.memit_seq_main import apply_memit_seq_to_model
from nse import NSEHyperParams
from nse.nse_main import apply_nse_to_model
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.globals import DATA_DIR, HPARAMS_DIR, KV_DIR, RESULTS_DIR


ALG_DICT = {
    "AlphaEdit": (AlphaEditHyperParams, apply_AlphaEdit_to_model),
    "PMPh": (PMPhHyperParams, apply_PMPh_to_model),
    "MEMIT_seq": (MEMITHyperParams, apply_memit_seq_to_model),
    "MEMIT_prune": (MEMITHyperParams, apply_memit_to_model),  # Not supported in this script
    "MEMIT_rect": (MEMITHyperParams, apply_memit_rect_to_model),
    "NSE": (NSEHyperParams, apply_nse_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    "mquake": (MQUAKEDataset, compute_rewrite_quality_mquake),
}


def _json_default(x: Any):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, (torch.Tensor,)):
        return x.detach().cpu().tolist()
    return str(x)


def _make_run_dir(dir_name: str) -> Path:
    base = RESULTS_DIR / dir_name
    base.mkdir(parents=True, exist_ok=True)
    id_list = [
        int(str(x).split("_")[-1])
        for x in base.iterdir()
        if x.is_dir() and str(x).split("_")[-1].isnumeric()
    ]
    run_id = 0 if not id_list else max(id_list) + 1
    run_dir = base / f"run_{str(run_id).zfill(3)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _get_project(model, tok, layer: int, hparams) -> torch.Tensor:
    """
    K_0-only nullspace projection: P = U_small U_small^T, where singular values < threshold.
    """
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples,
        hparams.mom2_dtype,
        force_recompute=False,
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small = (S < threshold).nonzero(as_tuple=True)[0]
    if len(small) == 0:
        return torch.zeros_like(cov)
    return U[:, small] @ U[:, small].T



def _init_seq_state(
    model: AutoModelForCausalLM, tok: AutoTokenizer, alg_name: str, hparams
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[Optional[torch.Tensor]]]]:
    """
    Returns (cache_c, P, K_p) for sequential algorithms.
    """
    if alg_name not in {"AlphaEdit", "PMPh", "MEMIT_seq", "NSE"}:
        return None, None, None

    W_out = nethook.get_parameter(
        model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight"
    )

    # Keep the same shape logic as in experiments/sequential_retest.py.
    if getattr(hparams, "model_name", None) == "gpt2-xl":
        d = W_out.shape[0]
    else:
        d = W_out.shape[1] if W_out.ndim == 2 else W_out.shape[-1]

    cache_c = torch.zeros((len(hparams.layers), d, d), device="cpu")

    P = None
    K_p = None
    if alg_name == "AlphaEdit":
        P = torch.zeros((len(hparams.layers), d, d), device="cpu")
        for i, layer in enumerate(hparams.layers):
            # Always use the non-whitened nullspace projection based on K_0.
            P[i, :, :] = _get_project(model, tok, layer, hparams)
    elif alg_name == "PMPh":
        P = torch.zeros((len(hparams.layers), d, d), device="cpu")
        for i, layer in enumerate(hparams.layers):
            # Always use the non-whitened nullspace projection based on K_0.
            P[i, :, :] = _get_project(model, tok, layer, hparams)
        # PMPh always uses the Schur-complement method, so always initialize K_p.
        K_p = [None] * len(hparams.layers)

    del W_out
    return cache_c, P, K_p


def _iter_requests(record_chunk: List[Dict]) -> List[Dict]:
    reqs: List[Dict] = []
    for record in record_chunk:
        rr = record["requested_rewrite"]
        rr_list = rr if isinstance(rr, list) else [rr]
        for rewrite_dict in rr_list:
            req = {"case_id": record["case_id"], **rewrite_dict}
            reqs.append(req)
    return reqs


def _maybe_snips_vec_for_case(
    case_id: int,
    snips: Optional[AttributeSnippets],
    vec,
    generation_test_interval: int,
):
    if snips is None or vec is None:
        return None, None
    if generation_test_interval <= 0:
        return None, None
    if case_id % generation_test_interval != 0:
        return None, None
    return snips, vec


def _eval_first_batch_checkpoint(
    *,
    checkpoint_idx: int,
    edited_upto_end_i: int,
    model,
    tok,
    ds,
    batch_size: int,
    ds_eval_method,
    snips,
    vec,
    generation_test_interval: int,
) -> Dict[str, Any]:
    """
    Under the current model state, run a full evaluation on the first batch
    (indices 0:batch_size) and return a dictionary of metrics for this checkpoint.
    """
    records = ds[:batch_size]
    metrics: List[Dict[str, Any]] = []
    for rec in records:
        case_id = rec["case_id"]
        cur_snips, cur_vec = _maybe_snips_vec_for_case(
            case_id, snips, vec, generation_test_interval
        )
        post = ds_eval_method(model, tok, rec, cur_snips, cur_vec)
        metrics.append(
            {
                "checkpoint_idx": checkpoint_idx,
                "case_id": case_id,
                "edited_upto_end_i": edited_upto_end_i,
                "post": post,
            }
        )

    return {
        "checkpoint_idx": checkpoint_idx,
        "edited_upto_end_i": edited_upto_end_i,
        "batch_size": batch_size,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a large-scale sequential editing experiment on ZSRE. "
            "By default, 5000 samples are edited in batches of size 100. "
            "After every 5 batches (i.e., every 500 samples), the first batch "
            "is re-evaluated, yielding 11 sets of batch-0 metrics "
            "(including the pre-edit baseline)."
        )
    )
    parser.add_argument(
        "--alg_name",
        choices=list(ALG_DICT.keys()),
        default="AlphaEdit",
        required=False,
        help=(
            "Editing algorithm to use; recommended to run this script separately "
            "for different methods such as AlphaEdit / PMPh / MEMIT_seq / NSE."
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name on Hugging Face, e.g., gpt2-xl / LLaMA.",
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        required=True,
        help="Name of the hyperparameter JSON file (located under HPARAMS_DIR/alg_name).",
    )
    parser.add_argument(
        "--ds_name",
        choices=list(DS_DICT.keys()),
        default="zsre",
        required=False,
        help="Dataset name, default is zsre; usually should be kept as zsre.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=5000,
        help="Maximum number of dataset samples to use, default 5000.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for each editing step, default 100.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=-1,
        help=(
            ">0: only perform generation tests for samples whose case_id is "
            "divisible by this value; <=0 disables generation tests."
        ),
    )
    parser.add_argument(
        "--skip_generation_tests",
        action="store_true",
        help="Force-disable generation tests (snips/vec are always None).",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Whether to use KV cache templates (only supported by some algorithms).",
    )
    parser.add_argument(
        "--dir_name",
        type=str,
        default=None,
        help="results/<dir_name>/run_xxx. Default: <alg_name>_zsre_large_sequential.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=0,
        help="CUDA device index to use, e.g., 0 / 1 / 2 ..., model is loaded to cuda:<cuda_device>.",
    )

    args = parser.parse_args()

    if args.alg_name == "MEMIT_prune":
        raise ValueError(
            "MEMIT_prune has extra global-matrix handling logic in experiments/evaluate.py; "
            "it is therefore not supported by this script."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    params_class, apply_algo = ALG_DICT[args.alg_name]
    params_path = HPARAMS_DIR / args.alg_name / args.hparams_fname
    hparams = params_class.from_json(params_path)

    run_dir = _make_run_dir(args.dir_name or f"{args.alg_name}_zsre_large_sequential")
    (run_dir / "retests").mkdir(parents=True, exist_ok=True)
    (run_dir / "edits").mkdir(parents=True, exist_ok=True)

    # Record the configuration of this experiment.
    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=_json_default)
    if not (run_dir / "params.json").exists():
        with open(params_path, "r") as src, open(run_dir / "params.json", "w") as dst:
            dst.write(src.read())

    device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    tok = AutoTokenizer.from_pretrained(args.model_name)
    tok.pad_token = tok.eos_token
    model.eval()

    ds_class, ds_eval_method = DS_DICT[args.ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=args.dataset_size_limit)
    total = len(ds)
    batch_size = args.batch_size
    num_batches = math.ceil(total / batch_size)

    # Initialize snips / vec only when generation tests are required.
    snips = None
    vec = None
    if (not args.skip_generation_tests) and args.generation_test_interval > 0:
        snips = AttributeSnippets(DATA_DIR)
        vec = get_tfidf_vectorizer(DATA_DIR)

    cache_template = None
    if args.use_cache:
        cache_template = (
            KV_DIR
            / f"{model.config._name_or_path.replace('/', '_')}_{args.alg_name}"
            / f"{args.ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )

    cache_c, P, K_p = _init_seq_state(model, tok, args.alg_name, hparams)

    edited_case_ids: List[int] = []

    # ---- First run a baseline evaluation on the first batch (before any edits). ----
    batch0_checkpoints: List[Dict[str, Any]] = []
    baseline_ckpt = _eval_first_batch_checkpoint(
        checkpoint_idx=0,
        edited_upto_end_i=0,
        model=model,
        tok=tok,
        ds=ds,
        batch_size=batch_size,
        ds_eval_method=ds_eval_method,
        snips=snips,
        vec=vec,
        generation_test_interval=args.generation_test_interval,
    )
    batch0_checkpoints.append(baseline_ckpt)

    # After every 5 batches (i.e., 500 samples), re-evaluate batch 0.
    checkpoint_every_batches = 5
    next_checkpoint_idx = 1  # 0 is already reserved as the baseline

    for batch_idx in range(num_batches):
        start_i = batch_idx * batch_size
        end_i = min((batch_idx + 1) * batch_size, total)
        record_chunk = ds[start_i:end_i]
        batch_case_ids = [r["case_id"] for r in record_chunk]
        edited_case_ids.extend(batch_case_ids)

        edit_marker = run_dir / "edits" / f"batch_{batch_idx:03d}_edited.json"
        retest_out = run_dir / "retests" / f"after_batch_{batch_idx:03d}.json"
        if retest_out.exists():
            # If this batch has already been edited and retested, skip it
            # (this makes it easy to resume from checkpoints).
            continue

        # ---- Apply edits cumulatively in sequence ----
        t0 = time()
        requests = _iter_requests(record_chunk)

        cache_algs = {"ROME", "MEMIT", "MEMIT_rect", "AlphaEdit", "PMPh", "MEMIT_seq", "NSE"}
        etc_args = (
            dict(cache_template=cache_template)
            if (cache_template is not None and args.alg_name in cache_algs)
            else dict()
        )
        seq_args = (
            dict(cache_c=cache_c)
            if args.alg_name in {"AlphaEdit", "PMPh", "MEMIT_seq", "NSE"}
            else dict()
        )

        nc_args: Dict[str, Any] = {}
        if args.alg_name == "AlphaEdit":
            # Always pass in the projection matrices P based solely on K_0.
            nc_args = dict(P=P)
        elif args.alg_name == "PMPh":
            # PMPh always uses the Schur-complement method, so always pass P and K_p.
            nc_args = dict(P=P)
            seq_args["K_p"] = K_p

        out = apply_algo(
            model,
            tok,
            deepcopy(requests),
            hparams,
            **etc_args,
            **seq_args,
            **nc_args,
        )

        # Handle different algorithms' return formats.
        if isinstance(out, tuple):
            if args.alg_name == "AlphaEdit":
                if len(out) == 3:
                    model, cache_c, K_p = out
                elif len(out) == 2:
                    model, cache_c = out
                else:
                    model = out[0]
            elif args.alg_name == "PMPh":
                # PMPh always returns (model, cache_c, K_p) when using Schur-complement method.
                if len(out) == 3:
                    model, cache_c, K_p = out
                elif len(out) == 2:
                    model, cache_c = out
                else:
                    model = out[0]
            elif args.alg_name in {"MEMIT_seq", "NSE"}:
                if len(out) >= 2:
                    model, cache_c = out[0], out[1]
                else:
                    model = out[0]
            else:
                model = out[0]
        else:
            model = out

        edit_time_s = time() - t0
        torch.cuda.empty_cache()

        with open(edit_marker, "w") as f:
            json.dump(
                {
                    "batch_idx": batch_idx,
                    "start_i": start_i,
                    "end_i": end_i,
                    "batch_case_ids": batch_case_ids,
                    "edited_case_ids": edited_case_ids,
                    "edit_time_s": edit_time_s,
                },
                f,
                indent=2,
                default=_json_default,
            )

        # ---- Re-evaluate all edited samples with the current model
        #      (consistent with experiments/sequential_retest) ----
        t1 = time()
        retest_records = ds[:end_i]
        metrics: List[Dict[str, Any]] = []
        for rec in retest_records:
            case_id = rec["case_id"]
            cur_snips, cur_vec = _maybe_snips_vec_for_case(
                case_id, snips, vec, args.generation_test_interval
            )
            post = ds_eval_method(model, tok, rec, cur_snips, cur_vec)
            metrics.append(
                {
                    "batch_idx": batch_idx,
                    "case_id": case_id,
                    "retested_at_end_i": end_i,
                    "new_batch_case_ids": batch_case_ids,
                    "edited_case_ids": edited_case_ids,
                    "post": post,
                }
            )
        retest_time_s = time() - t1

        with open(retest_out, "w") as f:
            json.dump(
                {
                    "batch_idx": batch_idx,
                    "start_i": start_i,
                    "end_i": end_i,
                    "batch_case_ids": batch_case_ids,
                    "edited_case_ids": edited_case_ids,
                    "edit_time_s": edit_time_s,
                    "retest_time_s": retest_time_s,
                    "metrics": metrics,
                },
                f,
                indent=2,
                default=_json_default,
            )

        print(
            f"[batch {batch_idx:03d}] edited {start_i}:{end_i} "
            f"(cases={batch_case_ids[0]}..{batch_case_ids[-1]}), "
            f"edit_time={edit_time_s:.2f}s, retest_time={retest_time_s:.2f}s"
        )

        # ---- If enough batches are finished and it's a multiple of 5,
        #      perform a checkpoint evaluation on batch_0. ----
        finished_batches = batch_idx + 1  # number of batches completed so far
        if finished_batches % checkpoint_every_batches == 0:
            ckpt = _eval_first_batch_checkpoint(
                checkpoint_idx=next_checkpoint_idx,
                edited_upto_end_i=end_i,
                model=model,
                tok=tok,
                ds=ds,
                batch_size=batch_size,
                ds_eval_method=ds_eval_method,
                snips=snips,
                vec=vec,
                generation_test_interval=args.generation_test_interval,
            )
            batch0_checkpoints.append(ckpt)
            next_checkpoint_idx += 1

    # Write (up to) 11 batch_0 checkpoints into a separate file for post-processing.
    with open(run_dir / "batch0_checkpoints.json", "w") as f:
        json.dump(batch0_checkpoints, f, indent=2, default=_json_default)

    print(f"Done. Results saved to: {run_dir}")
    print(f"Batch-0 checkpoints saved to: {run_dir / 'batch0_checkpoints.json'}")


if __name__ == "__main__":
    main()


