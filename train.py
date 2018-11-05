from __future__ import unicode_literals, print_function, division

import os
import time
import argparse

import tensorflow as tf
import torch
from model import Model
from torch.nn.utils import clip_grad_norm_

from torch.optim import Adagrad

from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.utils import calc_running_avg_loss
from train_util import get_input_from_batch, get_output_from_batch
from torch.autograd import Variable

use_cuda = config.use_gpu and torch.cuda.is_available()

from rouge import Rouge

rouge = Rouge()
from data_util import data


def rouge_l_f(decoded, reference):
    score = rouge.get_scores(decoded, reference)[0]
    return score['rouge-l']['f']


class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        time.sleep(15)

        train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = tf.summary.FileWriter(train_dir)

    def save_model(self, running_avg_loss, iter):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def train_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        self.optimizer.zero_grad()

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)
        s_t_1_origin = s_t_1

        batch_size = batch.batch_size
        step_losses = []

        sample_idx = []
        sample_log_probs = Variable(torch.zeros(batch_size))
        baseline_idx = []

        for di in range(min(max_dec_len, config.max_dec_steps)):

            y_t_1 = dec_batch[:, di]  # Teacher forcing, shape [batch_size]
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                                                           encoder_outputs,
                                                                                           encoder_feature,
                                                                                           enc_padding_mask, c_t_1,
                                                                                           extra_zeros,
                                                                                           enc_batch_extend_vocab,
                                                                                           coverage, di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

            # sample
            if di == 0:  # use decoder input[0], which is <BOS>
                sample_t_1 = dec_batch[:, di]
                s_t_sample = s_t_1_origin
                c_t_sample = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))

            final_dist, s_t_sample, c_t_sample, attn_dist, p_gen, next_coverage = self.model.decoder(sample_t_1,
                                                                                                     s_t_sample,
                                                                                                     encoder_outputs,
                                                                                                     encoder_feature,
                                                                                                     enc_padding_mask,
                                                                                                     c_t_sample,
                                                                                                     extra_zeros,
                                                                                                     enc_batch_extend_vocab,
                                                                                                     coverage, di)
            # according to final_dist to sample
            # change sample_t_1
            dist = torch.distributions.Categorical(final_dist)
            sample_t_1 = Variable(dist.sample())
            # record sample idx
            sample_idx.append(sample_t_1)  # tensor list
            # compute sample probability
            sample_log_probs += torch.log(
                final_dist.gather(1, sample_t_1.view(-1, 1)))  # gather value along axis=1. given index

            # baseline
            if di == 0:  # use decoder input[0], which is <BOS>
                baseline_t_1 = dec_batch[:, di]
                s_t_sample = s_t_1_origin
                c_t_sample = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))

            final_dist, s_t_baseline, c_t_baseline, attn_dist, p_gen, next_coverage = self.model.decoder(baseline_t_1,
                                                                                                         s_t_baseline,
                                                                                                         encoder_outputs,
                                                                                                         encoder_feature,
                                                                                                         enc_padding_mask,
                                                                                                         c_t_baseline,
                                                                                                         extra_zeros,
                                                                                                         enc_batch_extend_vocab,
                                                                                                         coverage, di)
            # according to final_dist to get baseline
            # change baseline_t_1
            baseline_t_1 = torch.autograd.Variable(final_dist.max(1))  # get max value along axis=1
            # record baseline probability
            baseline_idx.append(baseline_t_1)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)

        # according to sample_idx and baseline_idx to compute RL loss
        # map sample/baseline_idx to string
        # compute rouge score
        # compute loss
        sample_idx = torch.stack(sample_idx, dim=1).squeeze()  # expect shape (batch_size, seq_len)
        baseline_idx = torch.stack(baseline_idx, dim=1).squeeze()
        rl_loss = torch.zeros(batch_size)
        for i in range(sample_idx.shape[0]):  # each example in a batch
            sample_y = data.outputids2words(sample_idx[i], self.vocab,
                                            (batch.art_oovs[i] if config.pointer_gen else None))
            baseline_y = data.outputids2words(baseline_idx[i], self.vocab,
                                              (batch.art_oovs[i] if config.pointer_gen else None))
            true_y = batch.original_abstracts[i]

            sample_score = rouge_l_f(sample_y, true_y)
            baseline_score = rouge_l_f(baseline_y, true_y)

            sample_score = Variable(sample_score)
            baseline_score = Variable(baseline_score)

            rl_loss[i] = baseline_score - sample_score
        rl_loss = rl_loss * sample_log_probs

        gamma = 0.9984
        loss = (1 - gamma) * loss + gamma * rl_loss

        loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def trainIters(self, n_iters, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        while iter < n_iters:
            batch = self.batcher.next_batch()
            loss = self.train_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            if iter % 100 == 0:
                self.summary_writer.flush()
            print_interval = 1000
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval,
                                                                           time.time() - start, loss))
                start = time.time()
            if iter % 5000 == 0:
                self.save_model(running_avg_loss, iter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_file_path",
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()

    train_processor = Train()
    train_processor.trainIters(config.max_iterations, args.model_file_path)
