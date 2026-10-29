SUPPORTED_DECODE_MODES = ("sana",)


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
    if has_gs_fusion:
        raise ValueError(
            f"{source} contains g_s_fusion weights, but this branch supports "
            "only decode_mode='sana'"
        )
