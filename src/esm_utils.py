"""ESM-2 model loading and utilities"""
import os
import torch
import esm
from safetensors.torch import load_file
from .sae import SparseAutoencoder


# SAE weights filename pattern
SAE_WEIGHTS_PATTERN = "esm2_plm1280_l{layer}_sae4096.safetensors"


def load_esm_local(model_path, device='cpu'):
    """
    load ESM-2-650M from local .pt file
    returns model, alphabet, batch_converter
    """
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


def load_sae(layer, sae_weights_dir, device='cpu'):
    """
    load sae for specific layer
    sae weights are in safetensors format
    """
    path = os.path.join(sae_weights_dir, SAE_WEIGHTS_PATTERN.format(layer=layer))
    sae = SparseAutoencoder(d_model=1280, d_hidden=4096, k=32)
    sae.load_state_dict(load_file(path), strict=False)
    sae.to(device).eval()
    return sae


def tokenize_seq(seq, batch_converter, device='cpu'):
    """tokenize a single sequence"""
    _, _, tokens = batch_converter([("seq", seq)])
    return tokens.to(device)


def get_logits_and_hidden(model, tokens, layer_idx):
    """
    get logits and hidden states at specific layer
    repr_layers uses ESM-2's internal layer indexing
    """
    with torch.no_grad():
        results = model(tokens, repr_layers=[layer_idx], return_contacts=False)
    logits = results["logits"]  # (1, L+2, vocab)
    hidden = results["representations"][layer_idx]  # (1, L+2, 1280)
    return logits, hidden
