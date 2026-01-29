import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import csv
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .PMPh_hparams import PMPhHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def apply_PMPh_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: PMPhHyperParams,
    cache_template: Optional[str] = None,
    cache_c=None,
    P=None,
    P_updated: bool = True,
    K_p=None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    PMPh editing algorithm: based on AlphaEdit, augmented with Schur-complement
    constraints, and controlled by ``ablate_kp`` to optionally ablate the
    constraint on protected keys ``K_p``.
    """

    # Update target and print info
    model_device = next(model.parameters()).device
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"PMPh request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to(model_device))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)

    for i, layer in enumerate(hparams.layers):
        print(f"\n\n[PMPh] LAYER {layer}\n")

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        targets = zs - cur_zs
        print("[PMPh] z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = layer_ks.size(1) // targets.size(1)
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers

        # PMPh always uses the Schur-complement scheme whenever K_p is provided.
        use_schur = K_p is not None
        # Ablation switch: only effective when ``use_schur`` is True.
        ablate_kp = getattr(hparams, "ablate_kp", False) and use_schur

        if use_schur:
            # Schur-complement method or its ablation variant.
            P_layer = P[i, :, :].to(model_device)  # [d, d]
            K_i = layer_ks.to(model_device)  # [d, num_new_keys]
            K_i_hat = P_layer @ K_i  # [d, num_new_keys] = P @ K_i

            if ablate_kp:
                # --- Ablate the constraint on K_p ---
                R_mat = resid.to(model_device)  # [d, num_new_keys]
                Gram = K_i_hat.T @ K_i_hat  # [num_new_keys, num_new_keys]
                Gram_reg = Gram + hparams.L2 * torch.eye(
                    Gram.shape[0],
                    dtype=Gram.dtype,
                    device=Gram.device,
                )
                try:
                    Gram_inv = torch.linalg.inv(Gram_reg)
                except Exception:
                    print(
                        "Warning: (K_i_hat^T K_i_hat) is singular, using pseudo-inverse in ablation mode"
                    )
                    Gram_inv = torch.linalg.pinv(Gram_reg)

                upd_matrix = R_mat @ Gram_inv @ K_i_hat.T  # [d, d]

                del R_mat, Gram, Gram_reg, Gram_inv, K_i_hat, K_i, P_layer
            else:
                # --- Schur-complement scheme with a constraint on K_p ---
                K_p_layer = (
                    K_p[i]
                    if K_p is not None and i < len(K_p) and K_p[i] is not None
                    else None
                )

                if K_p_layer is not None and K_p_layer.numel() > 0:
                    # K_p_layer: [d, num_protected_keys]
                    K_p_layer = K_p_layer.to(model_device)
                    K_p_hat = P_layer @ K_p_layer  # [d, num_protected_keys]

                    # Gram matrices
                    G_ii = K_i_hat.T @ K_i_hat  # [num_new_keys, num_new_keys]
                    G_ip = K_i_hat.T @ K_p_hat  # [num_new_keys, num_protected_keys]
                    G_pp = K_p_hat.T @ K_p_hat  # [num_protected_keys, num_protected_keys]
                    G_pi = G_ip.T

                    try:
                        G_pp_inv = torch.linalg.inv(
                            G_pp
                            + hparams.L2
                            * torch.eye(
                                G_pp.shape[0],
                                dtype=G_pp.dtype,
                                device=G_pp.device,
                            )
                        )
                    except Exception:
                        print("Warning: G_pp is singular, using pseudo-inverse")
                        G_pp_inv = torch.linalg.pinv(
                            G_pp
                            + hparams.L2
                            * torch.eye(
                                G_pp.shape[0],
                                dtype=G_pp.dtype,
                                device=G_pp.device,
                            )
                        )

                    Sigma = (
                        G_ii
                        + hparams.L2
                        * torch.eye(
                            G_ii.shape[0],
                            dtype=G_ii.dtype,
                            device=G_ii.device,
                        )
                        - G_ip @ G_pp_inv @ G_pi
                    )

                    R = resid.T.to(model_device)  # [num_new_keys, d_resid]
                    try:
                        Sigma_inv = torch.linalg.inv(Sigma)
                    except Exception:
                        print("Warning: Sigma is singular, using pseudo-inverse")
                        Sigma_inv = torch.linalg.pinv(Sigma)

                    Lambda = Sigma_inv @ R  # [num_new_keys, d_resid]
                    M = -G_pp_inv @ G_pi @ Lambda  # [num_protected_keys, d_resid]

                    upd_matrix = K_i_hat @ Lambda + K_p_hat @ M  # [d, d_resid]

                    del (
                        K_p_layer,
                        K_p_hat,
                        G_ii,
                        G_ip,
                        G_pp,
                        G_pi,
                        G_pp_inv,
                        Sigma,
                        Sigma_inv,
                        Lambda,
                        M,
                    )
                else:
                    # K_p is empty: fall back to the unconstrained (no K_p) form.
                    upd_matrix = torch.linalg.solve(
                        P_layer @ K_i @ K_i.T
                        + hparams.L2
                        * torch.eye(
                            K_i.shape[0],
                            dtype=torch.float,
                            device=K_i.device,
                        ),
                        P_layer @ K_i @ resid.T.to(K_i.device),
                    )

                del K_i_hat, K_i, P_layer
        else:
            # Non-Schur path: identical to the current non-Schur AlphaEdit version.
            P_layer = P[i, :, :].to(model_device)
            K_layer = layer_ks.to(model_device)
            if cache_c is not None and cache_c[i, :, :] is not None:
                cache_layer = cache_c[i, :, :].to(model_device)
                gram = K_layer @ K_layer.T + cache_layer
            else:
                gram = K_layer @ K_layer.T
            upd_matrix = torch.linalg.solve(
                P_layer @ gram
                + hparams.L2
                * torch.eye(
                    K_layer.shape[0],
                    dtype=torch.float,
                    device=K_layer.device,
                ),
                P_layer @ K_layer @ resid.T.to(K_layer.device),
            )

        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        print("[PMPh] orig norm", torch.linalg.norm(weights[weight_name]))
        print("[PMPh] upd norm", torch.linalg.norm(upd_matrix))
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix

        for x in [layer_ks, cur_zs, targets, upd_matrix]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    # Update cache_c and K_p
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i, :, :] += layer_ks.cpu() @ layer_ks.cpu().T

        if K_p is not None:
            layer_ks_cpu = layer_ks.cpu()  # [d, num_new_keys]
            if K_p[i] is None:
                K_p[i] = layer_ks_cpu
            else:
                K_p[i] = torch.cat([K_p[i], layer_ks_cpu], dim=1)
            print(f"[PMPh] K_p[{i}] updated: shape {K_p[i].shape}")

    print(f"[PMPh] Deltas successfully computed for {list(weights.keys())}")

    if K_p is not None:
        return model, cache_c, K_p
    else:
        return model, cache_c


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use. PMPh-specific implementation.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    model_device = next(model.parameters()).device
    key = (model_name, layer_name)

    print(f"[PMPh] Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    cov = COV_CACHE[key].to(model_device)
    return torch.inverse(cov) if inv else cov


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by PMPh does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"[PMPh] Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE


__all__ = ["PMPhHyperParams", "apply_PMPh_to_model", "get_cov"]
