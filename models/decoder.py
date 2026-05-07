import torch
import torch.nn as nn


class DecoderRNN(nn.Module):
    """
    LSTM Decoder for generating captions.
    """

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Args:
            embed_size (int): The size of the embedding vector.
            hidden_size (int): The number of features in the hidden state.
            vocab_size (int): The size of the vocabulary.
            num_layers (int): The number of recurrent layers.
        """
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        """
        Forward pass for training with teacher forcing.

        Args:
            features (torch.Tensor): Image features from the encoder (batch_size, embed_size).
            captions (torch.Tensor): Ground truth captions (batch_size, seq_length).

        Returns:
            torch.Tensor: Predicted logits for each word in the vocabulary.
        """
        # Remove the <end> token from captions for input
        captions = captions[:, :-1]
        embeddings = self.embed(captions)

        # Prepend image features to the caption embeddings
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)

        # Pass through LSTM
        hiddens, _ = self.lstm(inputs)

        # Pass through the linear layer
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        """
        Generate captions for inference (greedy decoding).

        Args:
            inputs (torch.Tensor): Image features prepared for the LSTM (batch_size, 1, embed_size).
            states (tuple, optional): Initial hidden and cell states. Defaults to None.
            max_len (int, optional): Maximum length of the generated caption. Defaults to 20.

        Returns:
            tuple: (sampled_ids, avg_log_prob)
        """
        sampled_ids = []
        log_probs_sum = 0
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
            prob, predicted = log_probs.max(1)
            
            sampled_ids.append(predicted.item())
            log_probs_sum += prob.item()

            # Prepare the next input
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

            # Stop if <end> token is generated (<end> is index 2: <pad>=0, <start>=1, <end>=2, <unk>=3)
            if predicted.item() == 2:
                break
        
        avg_log_prob = log_probs_sum / len(sampled_ids) if len(sampled_ids) > 0 else 0
        return sampled_ids, avg_log_prob

    def sample_temperature(self, inputs, temperature=1.2, max_len=20, end_idx=2):
        """
        Sample next word from softmax(logits/temperature) instead of argmax.
        Higher temperature = more diverse (sometimes more relevant) words; can avoid always picking 'person'.
        
        Returns:
            tuple: (sampled_ids, avg_log_prob)
        """
        sampled_ids = []
        log_probs_sum = 0
        states = None
        device = inputs.device
        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            logits = self.linear(hiddens.squeeze(1))
            
            log_probs_orig = torch.nn.functional.log_softmax(logits, dim=1)
            
            if temperature <= 0:
                prob, predicted = log_probs_orig.max(1)
            else:
                probs = torch.softmax(logits / temperature, dim=1)
                predicted = torch.multinomial(probs, 1).squeeze(1)
                prob = log_probs_orig[0, predicted[0]]
                
            sampled_ids.append(predicted.item())
            log_probs_sum += prob.item()
            
            if predicted.item() == end_idx:
                break
            inputs = self.embed(predicted.unsqueeze(1))
            
        avg_log_prob = log_probs_sum / len(sampled_ids) if len(sampled_ids) > 0 else 0
        return sampled_ids, avg_log_prob

    def sample_beam(self, inputs, beam_size=5, max_len=20, end_idx=2, length_penalty=0.8, return_all=False, repetition_penalty=1.2):
        """
        Generate captions using length-normalized beam search with repetition penalty.

        Args:
            inputs (torch.Tensor): Image features (batch_size, 1, embed_size).
            beam_size (int): Number of beams to keep at each step.
            max_len (int): Maximum caption length.
            end_idx (int): Vocabulary index of <end> token.
            length_penalty (float): Score = log_prob / (length ** length_penalty).
            return_all (bool): If True, returns list of (ids, score) for all top beams.
            repetition_penalty (float): Penalty for repeating words. > 1.0 reduces repetition.

        Returns:
            tuple: (best_sampled_ids, best_norm_score) if return_all=False
            list: [(ids, score), ...] if return_all=True
        """
        batch_size = inputs.size(0)
        device = inputs.device

        def norm_score(log_p_sum, length):
            if length <= 0:
                return log_p_sum
            return log_p_sum / (length ** length_penalty)

        # beams: (log_prob_sum, word_ids_list, hidden_state_tuple)
        hidden = None
        hiddens, hidden = self.lstm(inputs, hidden)
        outputs = self.linear(hiddens.squeeze(1))
        log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
        
        # Apply initial repetition penalty if needed (though unlikely at start)
        topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=1)
        beams = []
        for k in range(beam_size):
            word_id = topk_ids[0, k].item()
            log_p = topk_log_probs[0, k].item()
            beams.append((log_p, [word_id], hidden))
        
        if beams[0][1][0] == end_idx:
            if return_all:
                return [(b[1], norm_score(b[0], 1)) for b in beams]
            return beams[0][1], norm_score(beams[0][0], 1)
            
        for _ in range(max_len - 1):
            all_candidates = []
            for log_p_sum, word_list, (h, c) in beams:
                if word_list[-1] == end_idx:
                    all_candidates.append((log_p_sum, word_list, (h, c)))
                    continue
                
                next_word = word_list[-1]
                next_input = self.embed(torch.tensor([[next_word]], device=device).long())
                hiddens, new_hidden = self.lstm(next_input, (h, c))
                outputs = self.linear(hiddens.squeeze(1))
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for word_id in set(word_list):
                        if outputs[0, word_id] > 0:
                            outputs[0, word_id] /= repetition_penalty
                        else:
                            outputs[0, word_id] *= repetition_penalty
                            
                log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
                topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=1)
                for k in range(beam_size):
                    w = topk_ids[0, k].item()
                    lp = topk_log_probs[0, k].item()
                    new_list = word_list + [w]
                    all_candidates.append((log_p_sum + lp, new_list, new_hidden))
            
            all_candidates.sort(key=lambda x: norm_score(x[0], len(x[1])), reverse=True)
            beams = all_candidates[:beam_size]
            if all(b[1][-1] == end_idx for b in beams):
                break
        
        if return_all:
            return [(b[1], norm_score(b[0], len(b[1]))) for b in beams]
        best = beams[0]
        return best[1], norm_score(best[0], len(best[1]))
