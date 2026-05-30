import torch


def get_optimizer(
    model,
    learning_rate,
    weight_decay,
    optimizer_type="AdamW",
    beta1=0.9,
    beta2=0.999,
    momentum=0.9,
):
    """
    Configure optimizer with weight decay, following nanoGPT patterns.

    Args:
        model: PyTorch model
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        optimizer_type: 'AdamW' or 'SGD'
        beta1, beta2: Adam betas
        momentum: SGD momentum

    Supported optimizers:
    - AdamW: Adaptive learning with decoupled weight decay
    - SGD: Stochastic gradient descent with momentum and Nesterov
    """
    # Define parameter groups for weight decay
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    # Parameters with weight decay (2D+ tensors like weights)
    decay = {pn for pn, p in param_dict.items() if p.dim() >= 2}
    # Parameters without weight decay (1D tensors like biases, LayerNorm)
    nodecay = {pn for pn, p in param_dict.items() if p.dim() < 2}

    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(nodecay))],
            "weight_decay": 0.0,
        },
    ]
    print(f"Decay params: {len(decay)}, No-decay params: {len(nodecay)}")

    if optimizer_type == "AdamW":
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2))
    elif optimizer_type == "SGD":
        return torch.optim.SGD(
            optim_groups, lr=learning_rate, momentum=momentum, nesterov=True
        )
    else:
        raise ValueError(
            f"Unsupported optimizer type: {optimizer_type}. Supported: 'AdamW', 'SGD'"
        )


def get_scheduler(
    optimizer,
    num_training_steps,
    scheduler_type="PolyLR",
    learning_rate=3e-4,
    gamma=0.9,
):
    """
    Create learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        num_training_steps: Total number of training steps
        scheduler_type: 'PolyLR', 'OneCycleLR', or 'MultiStepLR'
        learning_rate: Base learning rate (needed for OneCycleLR)
        gamma: Decay factor for PolyLR/MultiStepLR
    """
    if scheduler_type == "PolyLR":
        from torch.optim.lr_scheduler import PolynomialLR

        return PolynomialLR(optimizer, total_iters=num_training_steps, power=gamma)
    elif scheduler_type == "OneCycleLR":
        from torch.optim.lr_scheduler import OneCycleLR

        return OneCycleLR(
            optimizer, max_lr=learning_rate, total_steps=num_training_steps
        )
    elif scheduler_type == "MultiStepLR":
        from torch.optim.lr_scheduler import MultiStepLR

        # Convert milestone fractions to actual steps
        milestones = [int(frac * num_training_steps) for frac in [0.5, 0.75]]
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
