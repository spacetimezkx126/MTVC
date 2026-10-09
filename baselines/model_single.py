"""
LightQuant single-stock price models (LSTM / ALSTM / BiLSTM / Adv-ALSTM / DTML).

LSTM/ALSTM/BiLSTM from LightQuant; Adv-ALSTM from Adv-ALSTM-master; DTML from DTML-pytorch-main.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        batch_first: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.rnn = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=batch_first, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.bn = nn.BatchNorm1d(hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)
        c0 = torch.zeros_like(h0)
        out, _ = self.rnn(x, (h0, c0))
        out = out[:, -1, :] if self.batch_first else out[-1, :, :]
        out = self.bn(out)
        return torch.sigmoid(self.linear(out))


class ALSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        dropout: float,
        batch_first: bool,
        attention_size: int,
    ):
        super().__init__()
        self.encoder_rnn = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout,
        )
        self.encoder_bn = nn.BatchNorm1d(hidden_size)
        self.decoder_rnn = nn.LSTM(
            hidden_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout,
        )
        self.decoder_bn = nn.BatchNorm1d(hidden_size)
        self.attention = nn.Linear(hidden_size * 2, attention_size)
        self.attention_score = nn.Linear(attention_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_size, output_size)
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

    def _attention_layer(self, decoder_hidden, encoder_outputs):
        dec = decoder_hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        energy = torch.tanh(self.attention(torch.cat((dec, encoder_outputs), dim=2)))
        attn_weights = F.softmax(self.attention_score(energy).squeeze(2), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs, (hidden, cell) = self.encoder_rnn(x)
        encoder_outputs = self.encoder_bn(encoder_outputs.transpose(1, 2)).transpose(1, 2)
        context = self._attention_layer(hidden[-1], encoder_outputs)
        decoder_output, _ = self.decoder_rnn(context.unsqueeze(1), (hidden, cell))
        decoder_output = self.decoder_bn(decoder_output.transpose(1, 2)).transpose(1, 2)
        decoder_output = self.dropout(decoder_output)
        return torch.sigmoid(self.fc_out(decoder_output.squeeze(1)))


class BiLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 128,
        num_layers: int = 3,
        output_size: int = 1,
        dropout: float = 0.3,
        batch_first: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.rnn = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=batch_first, dropout=dropout, bidirectional=True,
        )
        self.bn = nn.BatchNorm1d(hidden_size * 2)
        self.dropout_layer = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_size * 2, 128)
        self.linear2 = nn.Linear(128, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)
        c0 = torch.zeros_like(h0)
        out, _ = self.rnn(x, (h0, c0))
        out = out[:, -1, :] if self.batch_first else out[-1, :, :]
        out = self.bn(out)
        out = self.dropout_layer(out)
        out = F.relu(self.linear1(out))
        out = self.dropout_layer(out)
        return torch.sigmoid(self.linear2(out))


class AdvALSTM(nn.Module):
    """
    Adv-ALSTM (Attentive LSTM + adversarial training on latent features).
    PyTorch port of Adv-ALSTM pred_lstm.py (att=1, adv=1).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        adv_eps: float = 0.01,
        adv_beta: float = 0.01,
        l2_alpha: float = 0.01,
    ):
        super().__init__()
        self.adv_eps = adv_eps
        self.adv_beta = adv_beta
        self.l2_alpha = l2_alpha
        self.in_fc = nn.Linear(input_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.att_W = nn.Linear(hidden_size, hidden_size)
        self.att_u = nn.Linear(hidden_size, 1, bias=False)
        self.out_fc = nn.Linear(hidden_size * 2, 1)
        for module in (self.in_fc, self.att_W, self.att_u, self.out_fc):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return fea_con: concat(last hidden state, attention context)."""
        mapped = torch.tanh(self.in_fc(x))
        outputs, _ = self.lstm(mapped)
        a_laten = torch.tanh(self.att_W(outputs))
        a_scores = self.att_u(a_laten).squeeze(-1)
        a_alphas = F.softmax(a_scores, dim=1)
        a_con = torch.sum(outputs * a_alphas.unsqueeze(-1), dim=1)
        return torch.cat([outputs[:, -1, :], a_con], dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        fea_con = self.encode(x)
        return torch.sigmoid(self.out_fc(fea_con).squeeze(-1))

    def _weighted_bce(
        self,
        prediction: torch.Tensor,
        labels: torch.Tensor,
        pos_weight: float | None,
    ) -> torch.Tensor:
        if pos_weight is None or pos_weight <= 0:
            return F.binary_cross_entropy(prediction, labels)
        weight = torch.where(
            labels >= 0.5,
            torch.full_like(labels, pos_weight),
            torch.ones_like(labels),
        )
        return F.binary_cross_entropy(prediction, labels, weight=weight)

    def _l2_out_fc(self) -> torch.Tensor:
        return self.out_fc.weight.pow(2).sum() + self.out_fc.bias.pow(2).sum()

    def compute_loss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        pos_weight: float | None = None,
    ) -> torch.Tensor:
        fea_con = self.encode(x)
        clean_pred = torch.sigmoid(self.out_fc(fea_con).squeeze(-1))
        clean_loss = self._weighted_bce(clean_pred, labels, pos_weight)

        grad = torch.autograd.grad(
            clean_loss, fea_con, retain_graph=True, create_graph=True,
        )[0]
        delta = F.normalize(grad, p=2, dim=1)
        adv_fea = fea_con + self.adv_eps * delta
        adv_pred = torch.sigmoid(self.out_fc(adv_fea).squeeze(-1))
        adv_loss = self._weighted_bce(adv_pred, labels, pos_weight)
        return clean_loss + self.l2_alpha * self._l2_out_fc() + self.adv_beta * adv_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)


class TimeAxisAttention(nn.Module):
    """Time-axis LSTM + attention (DTML-pytorch-main/notebooks/dtml.ipynb)."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=False,
        )
        self.lnorm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, rt_attn: bool = False):
        # x: (batch, window, features)
        o, (h, _) = self.lstm(x)
        # dtml.ipynb: score = bmm(o, h.permute(1, 2, 0)); softmax over window dim
        score = torch.bmm(o, h.permute(1, 2, 0))
        tx_attn = F.softmax(score, dim=1).squeeze(-1)
        context = torch.bmm(tx_attn.unsqueeze(1), o).squeeze(1)
        normed_context = self.lnorm(context)
        if rt_attn:
            return normed_context, tx_attn
        return normed_context, None


class DataAxisAttention(nn.Module):
    """Data-axis multi-head attention (DTML-pytorch-main/notebooks/dtml.ipynb)."""

    def __init__(self, hidden_size: int, n_heads: int, drop_rate: float = 0.1):
        super().__init__()
        self.multi_attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.lnorm1 = nn.LayerNorm(hidden_size)
        self.lnorm2 = nn.LayerNorm(hidden_size)
        self.drop_out = nn.Dropout(drop_rate)

    def forward(
        self,
        hm: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        rt_attn: bool = False,
    ):
        # hm: (batch, num_stocks, hidden)
        residual = hm
        hm_hat, dx_attn = self.multi_attn(
            hm, hm, hm, key_padding_mask=key_padding_mask,
        )
        hm_hat = self.lnorm1(residual + self.drop_out(hm_hat))
        residual = hm_hat
        hp = torch.tanh(hm + hm_hat + self.mlp(hm + hm_hat))
        hp = self.lnorm2(residual + self.drop_out(hp))
        if rt_attn:
            return hp, dx_attn
        return hp, None


class DTML(nn.Module):
    """
    DTML from upstream DTML-pytorch notebook

    Input: [B, seq_len, n_stocks, features]  (same HLC as LSTM, stacked by date)
    Index: one stock (--dtml_market_index>=0) or cross-section mean (default -1)
    Output: probabilities [B, n_stocks]
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        beta: float = 0.1,
        drop_rate: float = 0.1,
        market_index: int = -1,
    ):
        super().__init__()
        self.beta = beta
        self.market_index = market_index
        self.txattention = TimeAxisAttention(input_dim, hidden_dim, num_layers)
        self.dxattention = DataAxisAttention(hidden_dim, num_heads, drop_rate)
        self.linear = nn.Linear(hidden_dim, 1)

    def _index_sequence(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return index series (B, T, F); cross-section mean uses valid slots only."""
        if self.market_index >= 0:
            idx = min(self.market_index, x.size(2) - 1)
            return x[:, :, idx, :]
        if valid_mask is None:
            return x.mean(dim=2)
        w = valid_mask.unsqueeze(1).unsqueeze(-1).float()
        summed = (x * w).sum(dim=2)
        count = w.sum(dim=2).clamp(min=1.0)
        return summed / count

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        rt_attn: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, num_stocks, n_feat = x.size()
        # stocks: (B, N, T, F) -> (B*N, T, F)
        stocks = x.permute(0, 2, 1, 3).reshape(batch_size * num_stocks, seq_len, n_feat)
        c_stocks, _ = self.txattention(stocks, rt_attn=rt_attn)
        c_stocks = c_stocks.view(batch_size, num_stocks, -1)

        index_seq = self._index_sequence(x, valid_mask=valid_mask)
        c_index, _ = self.txattention(index_seq, rt_attn=rt_attn)
        c_index = c_index.unsqueeze(1)

        hm = c_stocks + self.beta * c_index
        attn_mask = None
        if valid_mask is not None:
            attn_mask = ~valid_mask.bool()
        hp, _ = self.dxattention(hm, key_padding_mask=attn_mask, rt_attn=rt_attn)
        return self.linear(hp).squeeze(-1)

    def predict(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(x, valid_mask=valid_mask))


SINGLE_MODEL_CHOICES = ("lstm", "alstm", "bi_lstm", "adv_alstm", "dtml")


def build_single_model(model_name: str, **kwargs) -> nn.Module:
    name = model_name.lower()
    input_size = int(kwargs.get("input_size", 3))
    hidden_size = int(kwargs.get("hidden_size", 128))
    num_layers = int(kwargs.get("num_layers", 2))
    dropout = float(kwargs.get("dropout", 0.1))
    batch_first = bool(kwargs.get("batch_first", True))
    attention_size = int(kwargs.get("attention_size", 128))
    n_heads = int(kwargs.get("n_heads", 4))

    if name == "lstm":
        return LSTM(input_size, hidden_size, num_layers, 1, dropout, batch_first)
    if name == "alstm":
        return ALSTM(input_size, hidden_size, 1, num_layers, dropout, batch_first, attention_size)
    if name == "bi_lstm":
        return BiLSTM(input_size, hidden_size, num_layers, 1, dropout, batch_first)
    if name == "adv_alstm":
        return AdvALSTM(
            input_size,
            hidden_size,
            adv_eps=float(kwargs.get("adv_eps", 0.01)),
            adv_beta=float(kwargs.get("adv_beta", 0.01)),
            l2_alpha=float(kwargs.get("adv_l2_alpha", 0.01)),
        )
    if name == "dtml":
        beta = float(kwargs.get("dtml_beta", 0.1))
        market_index = int(kwargs.get("dtml_market_index", -1))
        drop_rate = float(kwargs.get("dtml_drop_rate", 0.1))
        dtml_layers = int(kwargs.get("dtml_layers", num_layers))
        dtml_hidden = int(kwargs.get("dtml_hidden", 64))
        return DTML(
            input_dim=input_size,
            hidden_dim=dtml_hidden,
            num_heads=n_heads,
            num_layers=dtml_layers,
            beta=beta,
            drop_rate=drop_rate,
            market_index=market_index,
        )
    raise ValueError(f"Unknown model: {model_name}. Choose from {SINGLE_MODEL_CHOICES}")
