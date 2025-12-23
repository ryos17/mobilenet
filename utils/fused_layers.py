import torch
import torch.nn as nn


def fuse_conv_and_bn(conv, bn, bias=True):
    """Fuse Conv2d and BatchNorm2d into a single Conv2d layer.
    
    Args:
        conv: nn.Conv2d layer
        bn: nn.BatchNorm2d layer (can have affine=False)
        bias: bool, whether to add bias to the fused convolution
    
    Returns:
        Fused Conv2d layer
    """
    # Get device and dtype of the original conv layer
    device = conv.weight.device
    
    # Initialize fused convolution on the same device
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=bias  # No bias 
    ).to(device)
    
    # Prepare filters
    w_conv = conv.weight.data.clone().view(conv.out_channels, -1)
    
    # Handle BN weight
    if bn.weight is not None:
        # with affine=True
        w_bn = torch.diag(bn.weight.data.div(torch.sqrt(bn.eps + bn.running_var.data)))
    else:
        # with affine=False
        w_bn = torch.diag(1.0 / torch.sqrt(bn.eps + bn.running_var.data))
    
    # Compute fused weights
    fused_weight = torch.mm(w_bn, w_conv).view(fused_conv.weight.size()).detach()
    fused_conv.weight.data.copy_(fused_weight)
    
    # Add bias if provided
    if bias:
        # Handle Conv bias
        if conv.bias is not None:
            b_conv = conv.bias.data
        else:
            b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device, dtype=conv.weight.dtype)
        
        # Handle BN bias
        if bn.bias is not None and bn.weight is not None:
            # with affine=True
            b_bn = bn.bias.data - bn.weight.data.mul(bn.running_mean.data).div(torch.sqrt(bn.running_var.data + bn.eps))
        else:
            # with affine=False
            b_bn = -bn.running_mean.data.div(torch.sqrt(bn.running_var.data + bn.eps))
        
        fused_conv.bias.data.copy_(torch.matmul(w_bn, b_conv) + b_bn)
    
    return fused_conv


def fuse_model(model):
    """Recursively fuse all Conv2d + BatchNorm2d pairs in a model.
    
    This function traverses Sequential modules and fuses Conv+BN pairs,
    replacing them with a single fused Conv2d layer.
    
    Args:
        model: PyTorch model to fuse
    
    Returns:
        Fused model (modified in-place)
    """
    def _fuse_sequential(seq):
        """Fuse Conv+BN pairs in a Sequential module."""
        fused_layers = []
        i = 0
        while i < len(seq):
            layer = seq[i]
            
            # If current layer is Conv2d, it's always followed by BatchNorm2d
            if isinstance(layer, nn.Conv2d):
                # Next layer must be BatchNorm2d (we only have BN after Conv)
                if i + 1 < len(seq) and isinstance(seq[i + 1], nn.BatchNorm2d):
                    # Fuse Conv + BN
                    fused_conv = fuse_conv_and_bn(layer, seq[i + 1])
                    fused_layers.append(fused_conv)
                    i += 2  # Skip both Conv and BN
                    
                    # Check if there's a ReLU after BN (common pattern) and preserve ReLU if it exists
                    if i < len(seq) and isinstance(seq[i], nn.ReLU):
                        fused_layers.append(seq[i])
                        i += 1
                else:
                    # This shouldn't happen, but handle gracefully
                    fused_layers.append(layer)
                    i += 1
            # If current layer is Sequential, recursively fuse it
            elif isinstance(layer, nn.Sequential):
                fused_layers.append(_fuse_sequential(layer))
                i += 1
            else:
                fused_layers.append(layer)
                i += 1
        
        return nn.Sequential(*fused_layers)
    
    # Fuse the model
    if hasattr(model, 'model') and isinstance(model.model, nn.Sequential):
        model.model = _fuse_sequential(model.model)
    
    return model
