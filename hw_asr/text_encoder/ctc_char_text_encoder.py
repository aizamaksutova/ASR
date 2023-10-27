from typing import List, NamedTuple
from collections import defaultdict
from pyctcdecode import build_ctcdecoder
import numpy as np

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, lm_path, alpha, beta, unigrams_list, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.hotwords = ['irresolution', 'roughly', 'george', 'tenability', 'completely', 'nottingham', 'northerners', 'vanderpool',
        'amazement', 'misdemeanor', 'hypocrite', 'simplified', 'amusement', 'sandford', 'dedalus', 'creatures', 'animosity']
        self.hotwords = [word.upper() for word in self.hotwords]
        labels = ['']
        labels += list("".join(self.alphabet).upper())
        
        self.decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=lm_path,
            alpha=alpha,
            beta=beta,
            unigrams=unigrams_list
        )

    def ctc_decode(self, inds: List[int]) -> str:
        last_tok = 0
        s_final = []
        for indices in inds:
            if indices == last_tok:
                continue
            if indices != 0:
                s_final.append(indices)
                
            last_tok = indices
        return ''.join([self.ind2char[int(ind)] for ind in s_final]).strip()

    def extend_merge(self, frame, state):
        new_state = defaultdict(float)
        for next_char_ind, next_char_prob in enumerate(frame):
            next_char = self.ind2char[next_char_ind]

            for (pref, last_char), pref_proba in state.items():
                if last_char == next_char or next_char == self.EMPTY_TOK:
                    new_pref = pref
                else:
                    new_pref = pref + next_char
            new_state[(new_pref, next_char)] += pref_proba * next_char_prob
        return new_state


    def truncate(self, state, beam_size):
        state_list = list(state.items())
        return dict(sorted(state_list, key=lambda x: x[1])[-beam_size:])


    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []
        state = {('', self.EMPTY_TOK): 1.0}
        for row in probs:
            state = self.extend_merge(row, state)
            state = self.truncate(state, beam_size)
        
        for hyp, prob in state.items():
            hypos.append(Hypothesis(hyp[0], prob))

        return sorted(hypos, key=lambda x: x.prob, reverse=True)
    

    def ctc_beam_search_withlm(self, probs, probs_length, beam_size):
        """
        Performs beam search with LM and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []
        lm_text = self.decoder.decode(
            probs[:probs_length, :].numpy(), 
            beam_size,
            hotwords=self.hotwords,
            hotword_weight=10.0)

        lm_text = lm_text.lower()

        hypos.append(Hypothesis(lm_text, probs[:probs_length, :].numpy()))
        
        return sorted(hypos, key=lambda x: x.prob, reverse=True)


    

