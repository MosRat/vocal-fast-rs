"""Check full bidirectional online attention independently with FP64 softmax."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hubert_attention import online_attention


class OnlineAttentionTests(unittest.TestCase):
    def test_strides_tails_and_extreme_logits(self):
        torch.manual_seed(67)
        torch.set_num_threads(1)
        for amplitude in [1,16]:
            q=torch.randn(2,19,3,14).transpose(1,2)[...,::2]*amplitude
            k=torch.randn(2,37,3,14).transpose(1,2)[...,::2]*amplitude
            v=torch.randn(2,37,3,13).transpose(1,2)
            reference=(torch.softmax(q.double()@k.double().transpose(-2,-1)*.125,-1)@v.double()).float()
            for qt,kt in [(1,1),(4,16),(16,32)]:
                actual=online_attention(q,k,v,.125,qt,kt)
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual,reference,atol=2e-4,rtol=2e-4)

    def test_invalid_shapes_and_precision(self):
        q=torch.ones(1,2,4,3)
        with self.assertRaises(ValueError): online_attention(q,q,q,key_tile=0)
        with self.assertRaises(ValueError): online_attention(q.half(),q.half(),q.half())
        with self.assertRaises(ValueError): online_attention(q,q,q[:,:,:2])


if __name__=="__main__":unittest.main()
