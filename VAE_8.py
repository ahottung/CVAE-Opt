"""
This code is based on https://github.com/mveres01/pytorch-drl4vrp/blob/master/model.py
"""

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim
import torch.nn.functional as F
import tsp
import cvrp


class Embedding(nn.Module):
    """Encodes the coordinate states using 1D Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Embedding, self).__init__()
        self.embed = nn.Linear(input_size, hidden_size)

    def forward(self, input_data):
        output_data = self.embed(input_data)
        return output_data


class Encoder(nn.Module):
    def __init__(self, instance_embedding, reference_embedding, encoder_attn, rnn, update_fn, search_space_size,
                 hidden_size):
        super(Encoder, self).__init__()
        self.instance_embedding = instance_embedding
        self.reference_embedding = reference_embedding
        self.encoder_attn = encoder_attn
        self.gru_decoder = nn.GRU(hidden_size * 2, hidden_size, 1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, search_space_size)
        self.fc2 = nn.Linear(hidden_size, search_space_size)
        self.rnn = rnn
        self.update_fn = update_fn

    def forward(self, instance, solution, instance_hidden, config):
        batch_size, sequence_size, input_size, = instance.size()
        reference_input = instance[torch.arange(batch_size), solution[:, 0], :].unsqueeze(1).detach()

        last_hh = None
        last_hh_2 = None
        reference_hidden = self.reference_embedding(reference_input)
        for j in range(1, solution.shape[1]):

            rnn_out, last_hh = self.rnn(reference_hidden, last_hh)

            # Given a summary of the output, find an  input context
            enc_attn = self.encoder_attn(instance_hidden, rnn_out)
            context = enc_attn.permute(0, 2, 1).bmm(instance_hidden)

            ptr = solution.t()[j].long()

            if self.update_fn is not None:
                instance = self.update_fn(instance, ptr.data)
                instance_hidden = self.instance_embedding(instance)

            reference_input = torch.gather(instance, 1, ptr.view(-1, 1, 1).expand(-1, 1, input_size))
            reference_hidden = self.reference_embedding(reference_input)

            rnn_input = torch.cat((reference_hidden, context), dim=2)
            rnn_out_2, last_hh_2 = self.gru_decoder(rnn_input, last_hh_2)

        mu = self.fc1(last_hh_2.squeeze(0))
        log_var = self.fc2(last_hh_2.squeeze(0))
        return self.reparameterise(mu, log_var), mu, log_var

    @staticmethod
    def reparameterise(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, hidden_size, 1), requires_grad=True))
        self.W = nn.Parameter(torch.zeros((1, 2 * hidden_size, 1 * hidden_size), requires_grad=True))

    def forward(self, instance_hidden, rnn_out):
        batch_size, _, hidden_size = instance_hidden.size()

        hidden = rnn_out.expand_as(instance_hidden)
        hidden = torch.cat((instance_hidden, hidden), 2)

        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, -1, -1)
        W = self.W.expand(batch_size, -1, -1)

        ret = torch.bmm(hidden, W)
        attns = torch.bmm(torch.relu(ret), v)
        attns = F.softmax(attns, dim=1)  # (batch, seq_len)
        return attns


class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, encoder_attn, rnn, hidden_size, search_space_size):
        super(Pointer, self).__init__()

        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(torch.zeros((1, hidden_size, 1)
                                          , requires_grad=True))
        self.W = nn.Parameter(torch.zeros((1, 2 * hidden_size, hidden_size)
                                          , requires_grad=True))

        # Used to compute a representation of the current decoder output
        self.fc1 = nn.Linear(2 * hidden_size + search_space_size, 2 * hidden_size)
        self.fc2 = nn.Linear(2 * hidden_size, hidden_size)
        self.encoder_attn = encoder_attn
        self.rnn = rnn

    def forward(self, instance_hidden, reference_hidden, Z, last_hh):
        rnn_out, last_hh = self.rnn(reference_hidden, last_hh)
        rnn_out = rnn_out

        # Given a summary of the output, find an  input context
        enc_attn = self.encoder_attn(instance_hidden, rnn_out)
        context = enc_attn.permute(0, 2, 1).bmm(instance_hidden)  # (B, 1, num_feats)

        fc_input = torch.cat((context.squeeze(1), Z, reference_hidden.squeeze(1)), dim=1)  # (B, num_feats, seq_len)
        fc_output = self.fc1(fc_input)
        fc_output = self.fc2(fc_output).unsqueeze(1)
        fc_output = fc_output.expand(-1, instance_hidden.size(1), -1)
        fc_output = torch.cat((instance_hidden, fc_output), dim=2)

        v = self.v.expand(instance_hidden.size(0), -1, -1)
        W = self.W.expand(instance_hidden.size(0), -1, -1)
        probs = torch.bmm(torch.tanh(torch.bmm(fc_output, W)), v).squeeze(2)
        return probs, last_hh


class Decoder(nn.Module):
    def __init__(self, instance_embedding, reference_embedding, encoder_attn, rnn, hidden_size,
                 search_space_size, mask_fn, update_fn):
        super(Decoder, self).__init__()

        # Define the encoder & decoder models
        self.pointer = Pointer(encoder_attn, rnn, hidden_size, search_space_size)
        self.instance_embedding = instance_embedding
        self.reference_embedding = reference_embedding
        self.encoder_attn = encoder_attn
        self.mask_fn = mask_fn
        self.update_fn = update_fn
        self.rnn = rnn

    def forward(self, instance, solution, Z, instance_hidden, config, teacher_forcing, last_hh_new=None):
        batch_size, sequence_size, input_size, = instance.size()
        reference_input = instance[torch.arange(batch_size), solution[:, 0], :].unsqueeze(1).detach()
        max_steps = sequence_size if self.mask_fn is None else 10000
        tour_idx, tour_logp, tour_prob = [solution[:, [0]]], [], []

        mask = torch.ones(batch_size, sequence_size, device=config.device)
        mask[torch.arange(batch_size), solution[:, 0]] = 0
        for j in range(1, max_steps):
            if not mask.byte().any():
                break

            reference_hidden = self.reference_embedding(reference_input)
            probs, last_hh_new = self.pointer(instance_hidden, reference_hidden, Z, last_hh_new)
            probs = F.softmax(probs + mask.log(), dim=1)
            if teacher_forcing:
                # Select the actions based on the training solutions (during training)
                ptr = solution.t()[j].long()
                t = mask[torch.arange(len(mask)), ptr]
                assert t.eq(1).all()
                logp = torch.log(probs[torch.arange(batch_size), ptr])
                _, predicted_ptr = torch.max(probs, 1)
                tour_idx.append(predicted_ptr.data.unsqueeze(1))
            else:
                # Select actions greedily (during the search)
                prob, ptr = torch.max(probs, 1)
                logp = prob.log()
                tour_idx.append(ptr.data.unsqueeze(1))

            # After visiting a node update the dynamic representation
            if self.update_fn is not None:
                instance = self.update_fn(instance, ptr.data)
                instance_hidden = self.instance_embedding(instance)

                # Since we compute the VRP in minibatches, some tours may have
                # number of stops. We force the vehicles to remain at the depot
                # in these cases, and logp := 0
                is_done = instance[:, 3, :].sum(1).eq(0).float()
                logp = logp * (1. - is_done)

            if self.mask_fn is not None:
                mask = self.mask_fn(mask, instance[:, :, 2:], ptr).detach()

            reference_input = torch.gather(instance, 1, ptr.view(-1, 1, 1).expand(-1, 1, input_size))
            tour_prob.append(probs)
            tour_logp.append(logp.unsqueeze(1))

        tour_idx = torch.cat(tour_idx, dim=1)
        tour_logp = torch.cat(tour_logp, dim=1)

        return None, tour_idx, tour_logp


class VAE_8(nn.Module):
    def __init__(self, config):
        super(VAE_8, self).__init__()
        if config.problem == "TSP":
            input_size = 2
            mask_fn = tsp.update_mask
            update_fn = None
        elif config.problem == "CVRP":
            input_size = 4
            mask_fn = cvrp.update_mask
            update_fn = cvrp.update_dynamic

        hidden_size = 128
        self.instance_embedding = Embedding(input_size, hidden_size)
        reference_embedding = Embedding(input_size, hidden_size)
        encoder_attn = Attention(hidden_size)
        rnn = nn.GRU(hidden_size, hidden_size, 1, batch_first=True, dropout=0)

        self.encoder = Encoder(self.instance_embedding, reference_embedding, encoder_attn, rnn, update_fn,
                               config.search_space_size, hidden_size)
        self.decoder = Decoder(self.instance_embedding, reference_embedding, encoder_attn, rnn, hidden_size,
                               config.search_space_size, mask_fn, update_fn)

        self.instance_hidden = None
        self.dummy_solution = None

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, instance, solution_1, solution_2, config):
        instance_hidden = self.instance_embedding(instance)
        output_e = self.encoder(instance, solution_1, instance_hidden, config)

        Z, mu, log_var = output_e

        output_prob, tour_idx, tour_logp = self.decoder(instance, solution_2, Z, instance_hidden, config,
                                                        True)
        return output_prob, mu, log_var, Z, tour_idx, tour_logp

    def decode(self, instance, Z, config):
        if self.instance_hidden is None:
            self.instance_hidden = self.instance_embedding(instance)
        output_prob, tour_idx, tour_logp = self.decoder(instance, self.dummy_solution, Z, self.instance_hidden, config,
                                                        False)
        return output_prob, tour_idx, tour_logp

    def reset_decoder(self, batch_size, config):
        self.instance_hidden = None
        self.dummy_solution = torch.zeros(batch_size, 1).long().to(config.device)
