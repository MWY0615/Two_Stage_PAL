import os
import json
import torch
import logging
import difflib


def get_logger(name,
               format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
               date_format="%Y-%m-%d %H:%M:%S",
               file=False):
    """
    Get python logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # file or console
    handler = logging.StreamHandler() if not file else logging.FileHandler(name, mode='w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format_str,
                                  datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def dump_json(obj, fdir, name):
    """
    Dump python object in json
    """
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    with open(os.path.join(fdir, name), "w") as f:
        json.dump(obj, f, indent=4, sort_keys=False)


def load_json(fdir, name):
    """
    Load json as python object
    """
    path = os.path.join(fdir, name)
    if not os.path.exists(path):
        raise FileNotFoundError("Could not find json file: {}".format(path))
    with open(path, "r") as f:
        obj = json.load(f)
    return obj


def get_layer(l_name, library=torch.nn):
    """Return layer object handler from library e.g. from torch.nn

    E.g. if l_name=="elu", returns torch.nn.ELU.

    Args:
        l_name (string): Case insensitive name for layer in library (e.g. .'elu').
        library (module): Name of library/module where to search for object handler
        with l_name e.g. "torch.nn".

    Returns:
        layer_handler (object): handler for the requested layer e.g. (torch.nn.ELU)

    """
    all_torch_layers = [x for x in dir(torch.nn)]
    match = [x for x in all_torch_layers if l_name.lower() == x.lower()]
    if len(match) == 0:
        close_matches = difflib.get_close_matches(l_name,
                                                  [x.lower() for x in all_torch_layers])
        raise NotImplementedError("Layer with name {} not found in {}.\n Closest matches: {}".format(
            l_name, str(library), close_matches))

    elif len(match) > 1:
        close_matches = difflib.get_close_matches(l_name,
                                                  [x.lower() for x in all_torch_layers])
        raise NotImplementedError(
            "Multiple matchs for layer with name {} not found in {}.\n All matches: {}".format(
                l_name, str(library), close_matches))

    else:
        # valid
        layer_handler = getattr(library, match[0])
        return layer_handler


def load_obj(obj, device):
    """
    Offload tensor object in obj to cuda device
    """

    def cuda(obj): 
        return obj.to(device) if isinstance(obj, torch.Tensor) else obj

    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)
