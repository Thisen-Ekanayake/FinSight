# ═══════════════════════════════════════════════════════
# FinSight — Broker credential encryption
# ═══════════════════════════════════════════════════════
#
# Purpose : Encrypt and decrypt broker API keys with a server-held Fernet key.
#
# ══ WHY REFUSE RATHER THAN FALL BACK ══
#   With no BROKER_ENCRYPTION_KEY the only way to store a key would be in
#   plaintext. Other people's broker credentials in a plaintext SQLite file is
#   exactly the liability this module exists to avoid, so linking is simply
#   unavailable until the operator configures a key.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken

from src.broker import config


class EncryptionUnavailable(RuntimeError):
    """No usable BROKER_ENCRYPTION_KEY is configured."""


def _fernet() -> Fernet:
    if not config.BROKER_ENCRYPTION_KEY:
        raise EncryptionUnavailable("Broker linking is disabled on this deployment: BROKER_ENCRYPTION_KEY is not set.")
    try:
        return Fernet(config.BROKER_ENCRYPTION_KEY.encode())
    except (ValueError, TypeError) as exc:
        raise EncryptionUnavailable("BROKER_ENCRYPTION_KEY is not a valid Fernet key.") from exc


def encryption_available() -> bool:
    try:
        _fernet()
    except EncryptionUnavailable:
        return False
    return True


def encrypt(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    try:
        return _fernet().decrypt(ciphertext.encode()).decode()
    except InvalidToken as exc:
        raise EncryptionUnavailable(
            "Stored broker credentials cannot be decrypted — was BROKER_ENCRYPTION_KEY rotated? Re-link the account."
        ) from exc
