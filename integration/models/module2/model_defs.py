"""Auto-exported by kagglephase2.ipynb -- shared definitions for kagglephase3."""
import time, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

def initialize_weights(module):
    if isinstance(module, nn.GRU):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


class AdaptiveGRUModel(nn.Module):
    """Stacked GRU for variable-length windows (500-1000 steps, packed sequences).

    residual_indices: indices of the target columns within the input features.
    When set, forward() returns last_observed_value + correction, with the
    correction head zero-initialized so the model starts exactly at persistence.
    """

    def __init__(self, input_size=27, hidden_size=128, num_layers=2,
                 output_size=4, dropout=0.2, residual_indices=None):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.num_layers, self.output_size = num_layers, output_size
        self.residual_indices = residual_indices
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)
        self.apply(initialize_weights)
        if residual_indices is not None:
            self.register_buffer('_res_idx', torch.tensor(residual_indices, dtype=torch.long),
                                 persistent=False)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        out = self.fc2(self.relu(self.fc1(self.dropout(h_n[-1]))))
        if self.residual_indices is not None:
            last = x[torch.arange(x.size(0), device=x.device), lengths - 1]
            out = out + last[:, self._res_idx]
        return out

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class WindowDataset(Dataset):
    """Materializes windows per item from a memmapped (n_rows, n_feat) array.

    windows: (N, 2) int32 [anchor_end_exclusive, lookback_length].
    X = features[end-L:end]; y = features[end-1+horizon, target_idx].
    """

    def __init__(self, features_path, windows, horizon, target_idx):
        self.feat = np.load(features_path, mmap_mode='r')
        self.windows = windows
        self.h = int(horizon)
        self.tidx = np.asarray(target_idx)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        end, L = int(self.windows[i, 0]), int(self.windows[i, 1])
        x = torch.from_numpy(np.array(self.feat[end - L:end], dtype=np.float32))
        y = torch.from_numpy(np.array(self.feat[end - 1 + self.h, self.tidx], dtype=np.float32))
        return x, y, L


def collate_pad(batch):
    xs, ys, Ls = zip(*batch)
    T = max(Ls)
    X = torch.zeros(len(xs), T, xs[0].shape[1], dtype=torch.float32)
    for i, x in enumerate(xs):
        X[i, :x.shape[0]] = x
    return X, torch.stack(ys), torch.tensor(Ls, dtype=torch.long)


def make_loader(features_path, windows, horizon, target_idx, batch_size=64, shuffle=False):
    ds = WindowDataset(features_path, windows, horizon, target_idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      collate_fn=collate_pad, pin_memory=torch.cuda.is_available())

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False


def train_one_epoch(model, loader, optimizer, criterion, dev, grad_clip=1.0):
    model.train()
    total = 0.0
    for X, y, L in loader:
        X, y = X.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(X, L), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total += loss.item() * len(X)
    return total / len(loader.dataset)


def compute_metrics(preds_norm, targets_norm, target_std, target_mean, target_names):
    preds_real = preds_norm * target_std + target_mean
    targets_real = targets_norm * target_std + target_mean
    abs_err = np.abs(preds_real - targets_real)
    eps = 1e-6
    near0 = np.abs(targets_real) < eps
    ape = np.where(near0, np.nan, abs_err / np.where(near0, eps, np.abs(targets_real)) * 100.0)
    out = {}
    for i, name in enumerate(target_names):
        out[name] = {'mae': float(abs_err[:, i].mean()),
                     'rmse': float(np.sqrt(((preds_real[:, i] - targets_real[:, i]) ** 2).mean())),
                     'mape': float(np.nanmean(ape[:, i]))}
    out['mape_mean'] = float(np.nanmean([out[n]['mape'] for n in target_names]))
    return out


@torch.no_grad()
def predict_all(model, loader, dev):
    model.eval()
    preds, tgts = [], []
    for X, y, L in loader:
        preds.append(model(X.to(dev, non_blocking=True), L).cpu())
        tgts.append(y)
    return torch.cat(preds).numpy(), torch.cat(tgts).numpy()


def evaluate(model, loader, dev, target_std, target_mean, target_names):
    p, t = predict_all(model, loader, dev)
    m = compute_metrics(p, t, target_std, target_mean, target_names)
    m['loss'] = float(((p - t) ** 2).mean())
    return m


def train_model(model, train_loader, val_loader, dev, ckpt_path, epochs=30, lr=1e-3,
                patience=7, scheduler_patience=4, grad_clip=1.0, log_every=5):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=scheduler_patience, min_lr=1e-6)
    stopper = EarlyStopping(patience=patience)
    best = float('inf')
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, dev, grad_clip)
        with torch.no_grad():
            model.eval()
            v_tot = 0.0
            n = 0
            for X, y, L in val_loader:
                X, y = X.to(dev), y.to(dev)
                v_tot += nn.functional.mse_loss(model(X, L), y, reduction='sum').item()
                n += y.numel()
            vl = v_tot / n
        scheduler.step(vl)
        if stopper(vl):
            best = vl
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                        'val_loss': vl}, ckpt_path)
        if epoch == 1 or epoch % log_every == 0:
            print(f"  epoch {epoch:3d} | train={tr:.6f} | val={vl:.6f} | "
                  f"patience={stopper.counter}/{patience} | {time.time()-t0:.0f}s")
        if stopper.early_stop:
            print(f"  early stop at epoch {epoch}")
            break
    print(f"  done: best_val={best:.6f} | {time.time()-t0:.0f}s")
    return best

def persistence_preds(features, windows, target_idx):
    anchors = windows[:, 0] - 1
    return np.asarray(features[anchors][:, target_idx], dtype=np.float64)


def ses_fit_alpha(features, windows, horizon, target_idx, tail=240, max_n=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(windows), size=min(max_n, len(windows)), replace=False)
    alphas = np.round(np.arange(0.05, 1.0001, 0.05), 2)
    fitted = []
    for ti, col in enumerate(target_idx):
        segs = np.stack([np.array(features[max(e - tail, e - L):e, col], dtype=np.float64)[-tail:]
                         for e, L in windows[idx]])
        tgt = np.array([features[e - 1 + horizon, col] for e, L in windows[idx]], dtype=np.float64)
        best_a, best_err = 1.0, np.inf
        for a in alphas:
            w = a * (1.0 - a) ** np.arange(segs.shape[1] - 1, -1, -1)
            w /= w.sum()
            err = float(np.mean((segs @ w - tgt) ** 2))
            if err < best_err:
                best_a, best_err = a, err
        fitted.append(best_a)
    return fitted


def ses_preds(features, windows, target_idx, alphas, tail=240):
    out = np.empty((len(windows), len(target_idx)), dtype=np.float64)
    for ti, (col, a) in enumerate(zip(target_idx, alphas)):
        w = a * (1.0 - a) ** np.arange(tail - 1, -1, -1)
        w /= w.sum()
        segs = np.stack([np.array(features[e - tail:e, col], dtype=np.float64) for e, L in windows])
        out[:, ti] = segs @ w
    return out


class DriftMonitor:
    """EWMA of chunk-level error + z-score vs a frozen reference period."""

    def __init__(self, ewma_alpha=0.3, z_threshold=3.0, warmup_chunks=8, sustain=2):
        self.alpha, self.z, self.warmup, self.sustain = ewma_alpha, z_threshold, warmup_chunks, sustain
        self.ref = []
        self.ewma = None
        self.hits = 0
        self.drifting = False

    def update(self, chunk_error):
        self.ewma = chunk_error if self.ewma is None else             self.alpha * chunk_error + (1 - self.alpha) * self.ewma
        if len(self.ref) < self.warmup:
            self.ref.append(chunk_error)
            return False
        mu = float(np.mean(self.ref))
        sd = float(np.std(self.ref)) or 1e-12
        if (self.ewma - mu) / sd > self.z:
            self.hits += 1
        else:
            self.hits = 0
        self.drifting = self.hits >= self.sustain
        return self.drifting


class AdaptiveThreshold:
    """Confidence band from rolling error percentiles over recent chunks."""

    def __init__(self, window_chunks=20, lo_pct=50, hi_pct=90):
        self.window, self.lo, self.hi = window_chunks, lo_pct, hi_pct
        self.buf = []

    def update(self, chunk_abs_errors):
        self.buf.append(np.asarray(chunk_abs_errors))
        if len(self.buf) > self.window:
            self.buf.pop(0)
        allv = np.concatenate(self.buf)
        return float(np.percentile(allv, self.lo)), float(np.percentile(allv, self.hi))


class OnlineAdapter:
    """Error-triggered incremental fine-tune on the most recent seen windows."""

    def __init__(self, features_path, target_idx, horizon, lr=1e-4, recent=4096,
                 batch_size=64, epochs=1, grad_clip=1.0):
        self.features_path, self.tidx, self.h = features_path, target_idx, horizon
        self.lr, self.recent, self.bs, self.epochs, self.clip = lr, recent, batch_size, epochs, grad_clip
        self.n_updates = 0

    def adapt(self, model, seen_windows, dev):
        recent = seen_windows[-self.recent:]
        loader = make_loader(self.features_path, recent, self.h, self.tidx,
                             batch_size=self.bs, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        model.train()
        for _ in range(self.epochs):
            for X, y, L in loader:
                X, y = X.to(dev), y.to(dev)
                optimizer.zero_grad()
                loss = criterion(model(X, L), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip)
                optimizer.step()
        model.eval()
        self.n_updates += 1
        return model