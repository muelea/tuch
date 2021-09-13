import torch 

def to_tensors(dict, unsqueeze=False):
    if unsqueeze:
        for key, val in dict.items():
            try:
                dict[key] = torch.Tensor(val).unsqueeze(0)
            except TypeError:
                pass
    else:
        for key, val in dict.items():
            try:
                dict[key] = torch.Tensor(val)
            except TypeError:
                pass
    return dict