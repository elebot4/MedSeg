def get_mem_report(model, input_shape=None, optimizer_type="adam"):
    # Base parameter size (Weights)
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())

    # Static Inference Size
    inference_mb = (param_size + buffer_size) / 1024**2

    # Optimizer State Multiplier
    # SGD: 0 extra, SGD+Momentum: 1 extra, Adam: 2 extra
    opt_map = {"sgd": 0, "momentum": 1, "adam": 2, "adamw": 2}
    opt_multiplier = opt_map.get(optimizer_type.lower(), 2)

    # Calculation:
    # Weights (1) + Gradients (1) + Optimizer States (N)
    train_multiplier = 1 + 1 + opt_multiplier
    train_mb = (param_size * train_multiplier + buffer_size) / 1024**2

    return {
        "weights_only_mb": round(inference_mb, 2),
        "training_static_mb": round(train_mb, 2),
        "note": f"Training estimate assumes {optimizer_type} and excludes activations.",
    }
