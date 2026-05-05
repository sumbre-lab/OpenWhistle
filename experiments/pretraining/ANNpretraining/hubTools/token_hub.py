from __future__ import annotations

import os

import huggingface_hub


TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
nameaccount = (
    os.environ.get("HF_REPO_OWNER")
    or os.environ.get("HF_USERNAME")
    or os.environ.get("HF_NAMESPACE")
)


def login_to_hub(token: str | None = TOKEN) -> None:
    if token:
        huggingface_hub.login(token=token)


def require_hf_repo_owner(owner: str | None = nameaccount) -> str:
    if owner:
        return owner
    raise ValueError(
        "No Hugging Face repo owner configured. Set HF_REPO_OWNER, HF_USERNAME, or HF_NAMESPACE."
    )
