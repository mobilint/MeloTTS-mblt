"""MeloTTS text-to-speech for Mobilint NPUs.

``from melotts_mblt import TTS`` loads the speech synthesizer. The import is lazy, so ``import melotts_mblt`` does not
import ``torch`` or ``transformers``.
"""

from typing import TYPE_CHECKING

__version__ = "0.0.0"

if TYPE_CHECKING:
    from .api import TTS

__all__ = ["TTS", "__version__"]


def __getattr__(name: str):
    if name == "TTS":
        from .api import TTS

        return TTS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
