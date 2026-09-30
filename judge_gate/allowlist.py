"""Hard model-id allowlist for judge backends. Exactly these seven ids, no more."""

ALLOWLIST = frozenset({
    "XiaomiMiMo/MiMo-V2.6-Flash",
    "ByteDance/Seed-2.0-mini",
    "inclusionAI/Ling-3.0-flash",
    "meta-models/Muse-Glimmer-30B",
    "nvidia/Nemotron-3-Nano-30B-A3B",
    "thinkingmachines/Inkling-Small",
    "tencent/Hy3",
})


class ModelNotAllowlisted(ValueError):
    pass


def assert_allowed(model: str) -> None:
    if model not in ALLOWLIST:
        raise ModelNotAllowlisted(
            f"model {model!r} is not on the judge-gate allowlist "
            f"({len(ALLOWLIST)} ids; see judge_gate/allowlist.py)")
