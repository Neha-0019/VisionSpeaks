"""
LSTM Decoder with visual attention over spatial encoder features.
Use with EncoderCNNWithAttention; requires retraining.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Bahdanau-style attention: score = linear(hidden, encoder_outputs) then softmax."""

    def __init__(self, hidden_size, encoder_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size + encoder_size, 1)

    def forward(self, hidden, encoder_outputs):
        """
        hidden: (batch, hidden_size)
        encoder_outputs: (batch, num_locs, encoder_size)  [flattened spatial]
        Returns:
            context: (batch, encoder_size)
            weights: (batch, num_locs) for visualization
        """
        batch, num_locs, enc_size = encoder_outputs.size()
        hidden_exp = hidden.unsqueeze(1).expand(-1, num_locs, -1)  # (B, L, H)
        combined = torch.cat([hidden_exp, encoder_outputs], dim=2)  # (B, L, H+E)
        scores = self.attention(combined).squeeze(2)  # (B, L)
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, E)
        return context, weights


class DecoderRNNAttention(nn.Module):
    """
    LSTM Decoder that attends over spatial encoder features at each time step.
    """

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, encoder_size=None):
        super(DecoderRNNAttention, self).__init__()
        if encoder_size is None:
            encoder_size = embed_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = AttentionLayer(hidden_size, encoder_size)
        # Input to LSTM: concat of context (encoder_size) and word_embed (embed_size)
        self.lstm_input_size = encoder_size + embed_size
        self.lstm = nn.LSTM(self.lstm_input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)

    def forward(self, global_feat, spatial_feat, captions):
        """
        Training forward with teacher forcing.
        global_feat: (B, embed_size)
        spatial_feat: (B, embed_size, H, W)
        captions: (B, seq_len) includes <start> ... <end>
        """
        captions_in = captions[:, :-1]
        B, E, H, W = spatial_feat.size()
        encoder_outputs = spatial_feat.view(B, E, -1).permute(0, 2, 1)  # (B, H*W, E)
        seq_len = captions_in.size(1)
        h = self.init_h(global_feat).unsqueeze(0).expand(self.lstm.num_layers, B, -1)
        c = self.init_c(global_feat).unsqueeze(0).expand(self.lstm.num_layers, B, -1)
        outputs_list = []
        for t in range(seq_len):
            if t == 0:
                context = global_feat
            else:
                context, _ = self.attention(h[-1], encoder_outputs)
            word_embed = self.embed(captions_in[:, t])
            lstm_in = torch.cat([context, word_embed], dim=1).unsqueeze(1)
            out, (h, c) = self.lstm(lstm_in, (h, c))
            logits = self.linear(out.squeeze(1))
            outputs_list.append(logits)
        outputs = torch.stack(outputs_list, dim=1)
        return outputs

    def sample(self, global_feat, spatial_feat, max_len=20, end_idx=2):
        """Greedy decode with attention. For beam search, extend similarly.
        
        Returns:
            tuple: (sampled_ids, avg_log_prob)
        """
        B = global_feat.size(0)
        device = global_feat.device
        E, H, W = spatial_feat.size(1), spatial_feat.size(2), spatial_feat.size(3)
        encoder_outputs = spatial_feat.view(B, E, -1).permute(0, 2, 1)
        start_idx = 1
        h = self.init_h(global_feat).unsqueeze(0).expand(self.lstm.num_layers, B, -1)
        c = self.init_c(global_feat).unsqueeze(0).expand(self.lstm.num_layers, B, -1)
        sampled_ids = []
        log_probs_sum = 0
        word = torch.tensor([[start_idx]], device=device).long()
        for _ in range(max_len):
            if len(sampled_ids) == 0:
                context = global_feat
            else:
                context, _ = self.attention(h[-1], encoder_outputs)
            word_embed = self.embed(word.squeeze(1))
            lstm_in = torch.cat([context, word_embed], dim=1).unsqueeze(1)
            out, (h, c) = self.lstm(lstm_in, (h, c))
            logits = self.linear(out.squeeze(1))
            log_probs = torch.nn.functional.log_softmax(logits, dim=1)
            prob, word = log_probs.max(1)
            word = word.unsqueeze(1)
            sampled_ids.append(word.item())
            log_probs_sum += prob.item()
            if word.item() == end_idx:
                break
        
        avg_log_prob = log_probs_sum / len(sampled_ids) if len(sampled_ids) > 0 else 0
        return sampled_ids, avg_log_prob

    def sample_beam(self, global_feat, spatial_feat, beam_size=5, max_len=20, end_idx=2, length_penalty=0.8, return_all=False, repetition_penalty=1.2):
        """Beam search for attention decoder with repetition penalty.
        
        Returns:
            tuple: (best_sampled_ids, best_norm_score) if return_all=False
            list: [(ids, score), ...] if return_all=True
        """
        B = global_feat.size(0)
        device = global_feat.device
        E, H, W = spatial_feat.size(1), spatial_feat.size(2), spatial_feat.size(3)
        encoder_outputs = spatial_feat.view(B, E, -1).permute(0, 2, 1)
        start_idx = 1

        def norm_score(log_p_sum, length):
            return log_p_sum / ((length + 1e-6) ** length_penalty)

        h = self.init_h(global_feat).unsqueeze(0).expand(self.lstm.num_layers, B, -1)
        c = self.init_c(global_feat).unsqueeze(0).expand(self.lstm.num_layers, B, -1)
        context = global_feat
        word = torch.tensor([[start_idx]], device=device).long()
        word_embed = self.embed(word.squeeze(1))
        lstm_in = torch.cat([context, word_embed], dim=1).unsqueeze(1)
        out, (h, c) = self.lstm(lstm_in, (h, c))
        logits = self.linear(out.squeeze(1))
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=1)
        beams = []
        for k in range(beam_size):
            w = topk_ids[0, k].item()
            lp = topk_log_probs[0, k].item()
            beams.append((lp, [w], (h.clone(), c.clone())))
        
        if beams[0][1][0] == end_idx:
            if return_all:
                return [(b[1], norm_score(b[0], 1)) for b in beams]
            return beams[0][1], norm_score(beams[0][0], 1)
            
        for _ in range(max_len - 1):
            all_candidates = []
            for log_p_sum, word_list, (h_b, c_b) in beams:
                if word_list[-1] == end_idx:
                    all_candidates.append((log_p_sum, word_list, (h_b, c_b)))
                    continue
                context, _ = self.attention(h_b[-1], encoder_outputs)
                next_input = torch.tensor([[word_list[-1]]], device=device).long()
                word_embed = self.embed(next_input.squeeze(1))
                lstm_in = torch.cat([context, word_embed], dim=1).unsqueeze(1)
                out, (h_mid, c_mid) = self.lstm(lstm_in, (h_b, c_b))
                logits = self.linear(out.squeeze(1))
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for word_id in set(word_list):
                        if logits[0, word_id] > 0:
                            logits[0, word_id] /= repetition_penalty
                        else:
                            logits[0, word_id] *= repetition_penalty
                            
                log_probs = torch.nn.functional.log_softmax(logits, dim=1)
                topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=1)
                for k in range(beam_size):
                    w = topk_ids[0, k].item()
                    lp = topk_log_probs[0, k].item()
                    ctx, _ = self.attention(h_mid[-1], encoder_outputs)
                    w_embed = self.embed(torch.tensor([w], device=device).long())
                    lstm_in_w = torch.cat([ctx, w_embed], dim=1).unsqueeze(1)
                    _, (h_final, c_final) = self.lstm(lstm_in_w, (h_mid, c_mid))
                    all_candidates.append((log_p_sum + lp, word_list + [w], (h_final.clone(), c_final.clone())))
            all_candidates.sort(key=lambda x: norm_score(x[0], len(x[1])), reverse=True)
            beams = all_candidates[:beam_size]
            if all(b[1][-1] == end_idx for b in beams):
                break
        
        if return_all:
            return [(b[1], norm_score(b[0], len(b[1]))) for b in beams]
        best = beams[0]
        return best[1], norm_score(best[0], len(best[1]))