"""
utils/checkpoint.py — Checkpoint Save / Load / Auto-Resume
===========================================================

Designed with Kaggle's free-tier constraints in mind:
- Saves every 500 iterations (not just at end of training)
- Keeps the last 3 checkpoints to avoid filling disk quota
- Provides an emergency save callable for SIGTERM handlers
- Auto-resumes from the latest checkpoint if one exists
"""

import os
import glob
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class CheckpointManager:
    """Manages saving and loading of training checkpoints.

    On Kaggle free tier you have ~20 GB of disk in /kaggle/working.
    Each checkpoint is ~50 MB (two model states + optimiser states),
    so keeping 3 checkpoints uses ~150 MB — well within quota.

    Args:
        save_dir:    Directory to write checkpoints into.
        keep_last_n: Maximum number of checkpoints to retain. Older
                     checkpoints are automatically deleted.
    """

    def __init__(self, save_dir: str | Path, keep_last_n: int = 3) -> None:
        self.save_dir    = Path(save_dir)
        self.keep_last_n = keep_last_n
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Manifest file tracks all saved checkpoints in order
        self._manifest_path = self.save_dir / 'manifest.json'
        self._saved: list[str] = self._load_manifest()

    # ── Public API ────────────────────────────────────────────────────────────

    def save(
        self,
        G:           nn.Module,
        D:           nn.Module,
        G_ema:       nn.Module,
        opt_G:       torch.optim.Optimizer,
        opt_D:       torch.optim.Optimizer,
        iteration:   int,
        loss_g:      float = 0.0,
        loss_d:      float = 0.0,
        scaler_G:    Optional[object] = None,
        scaler_D:    Optional[object] = None,
        is_emergency: bool = False,
    ) -> Path:
        """Save all model and optimiser states.

        Args:
            is_emergency: When True, adds '_emergency' suffix so the file
                          is easily identifiable after a Kaggle timeout.

        Returns:
            Path of the saved checkpoint file.
        """
        tag   = f"iter_{iteration:07d}"
        if is_emergency:
            tag += "_emergency"
        fname = self.save_dir / f"ckpt_{tag}.pt"

        payload = {
            'iteration':   iteration,
            'G':           G.state_dict(),
            'D':           D.state_dict(),
            'G_ema':       G_ema.state_dict(),
            'opt_G':       opt_G.state_dict(),
            'opt_D':       opt_D.state_dict(),
            'loss_g':      loss_g,
            'loss_d':      loss_d,
        }
        if scaler_G is not None:
            payload['scaler_G'] = scaler_G.state_dict()
        if scaler_D is not None:
            payload['scaler_D'] = scaler_D.state_dict()

        torch.save(payload, fname)
        print(f"[Checkpoint] Saved: {fname.name}  (iter {iteration:,})")

        self._saved.append(str(fname))
        self._save_manifest()
        self._prune_old(is_emergency)

        return fname

    def load_latest(
        self,
        G:       nn.Module,
        D:       nn.Module,
        G_ema:   nn.Module,
        opt_G:   torch.optim.Optimizer,
        opt_D:   torch.optim.Optimizer,
        device:  torch.device,
        scaler_G: Optional[object] = None,
        scaler_D: Optional[object] = None,
    ) -> int:
        """Load the most recent checkpoint in-place.

        Returns:
            The iteration number stored in the checkpoint, or 0 if no
            checkpoint is found (fresh training).
        """
        ckpt_path = self._find_latest()
        if ckpt_path is None:
            print("[Checkpoint] No checkpoint found — starting from scratch.")
            return 0

        print(f"[Checkpoint] Resuming from: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)

        G.load_state_dict(payload['G'])
        D.load_state_dict(payload['D'])
        G_ema.load_state_dict(payload['G_ema'])
        opt_G.load_state_dict(payload['opt_G'])
        opt_D.load_state_dict(payload['opt_D'])

        if scaler_G is not None and 'scaler_G' in payload:
            scaler_G.load_state_dict(payload['scaler_G'])
        if scaler_D is not None and 'scaler_D' in payload:
            scaler_D.load_state_dict(payload['scaler_D'])

        iteration = payload['iteration']
        print(f"[Checkpoint] Resumed at iteration {iteration:,}  "
              f"(G_loss={payload.get('loss_g', 0):.4f}, "
              f"D_loss={payload.get('loss_d', 0):.4f})")
        return iteration

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _find_latest(self) -> Optional[Path]:
        """Return the most recently saved checkpoint path."""
        # Prefer manifest; fall back to glob scan
        if self._saved:
            for p in reversed(self._saved):
                if Path(p).exists():
                    return Path(p)

        # Glob fallback
        pattern = str(self.save_dir / 'ckpt_iter_*.pt')
        files   = sorted(glob.glob(pattern))
        return Path(files[-1]) if files else None

    def _prune_old(self, skip_emergency: bool = True) -> None:
        """Delete oldest checkpoints exceeding ``keep_last_n``."""
        existing = [p for p in self._saved if Path(p).exists()]
        # Never prune emergency checkpoints
        if skip_emergency:
            non_emerg = [p for p in existing if '_emergency' not in p]
        else:
            non_emerg = existing

        while len(non_emerg) > self.keep_last_n:
            oldest = non_emerg.pop(0)
            try:
                os.remove(oldest)
                print(f"[Checkpoint] Pruned: {Path(oldest).name}")
            except OSError:
                pass
            self._saved = [p for p in self._saved if p != oldest]

        self._save_manifest()

    def _load_manifest(self) -> list[str]:
        if self._manifest_path.exists():
            with open(self._manifest_path) as f:
                return json.load(f).get('checkpoints', [])
        return []

    def _save_manifest(self) -> None:
        with open(self._manifest_path, 'w') as f:
            json.dump({'checkpoints': self._saved}, f, indent=2)
