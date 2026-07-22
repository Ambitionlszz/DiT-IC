SUPPORTED_DECODE_MODES = ("sana", "gs_residual")


def validate_decode_mode(decode_mode):
    if decode_mode not in SUPPORTED_DECODE_MODES:
        raise ValueError(
            f"decode_mode must be one of {SUPPORTED_DECODE_MODES}, "
            f"got {decode_mode!r}"
        )
    return decode_mode


def validate_decode_checkpoint(
    decode_mode,
    has_gs_fusion,
    *,
    training,
    allow_missing_gs_fusion,
    source,
):
    validate_decode_mode(decode_mode)
    if decode_mode == "gs_residual" and not has_gs_fusion:
        if training and allow_missing_gs_fusion:
            return
        raise ValueError(
            f"{source} has no g_s_fusion weights, but decode_mode='gs_residual'"
        )
    if decode_mode == "sana" and has_gs_fusion:
        raise ValueError(
            f"{source} contains g_s_fusion weights, but decode_mode='sana'"
        )
