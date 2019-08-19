import torch as t
from utils import spectrogram2wav
from scipy.io.wavfile import write
import hyperparams as hp
from text import text_to_sequence
import numpy as np
from network import ModelPostNet, Model
from collections import OrderedDict
from tqdm import tqdm
import argparse

def load_checkpoint(step, model_name="transformer"):
    state_dict = t.load('/mnt/git/checkpoints/trans_tts/ch/checkpoint_%s_%d.pth.tar'% (model_name, step))   
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value

    return new_state_dict

def synthesis(text, args):
    m = Model()
    m_post = ModelPostNet()

    m.load_state_dict(load_checkpoint(args.restore_step1, "transformer"))
    m_post.load_state_dict(load_checkpoint(args.restore_step2, "postnet"))

    text = np.asarray(text_to_sequence(text, [hp.cleaners]))
    text = t.LongTensor(text).unsqueeze(0)
    text = text.cuda()
    mel_input = t.zeros([1,1, 80]).cuda()
    pos_text = t.arange(1, text.size(1)+1).unsqueeze(0)
    pos_text = pos_text.cuda()

    m=m.cuda()
    m_post = m_post.cuda()
    m.train(False)
    m_post.train(False)
    
    pbar = tqdm(range(args.max_len))
    with t.no_grad():
        for i in pbar:
            pos_mel = t.arange(1,mel_input.size(1)+1).unsqueeze(0).cuda()
            mel_pred, postnet_pred, attn, stop_token, _, attn_dec = m.forward(text, mel_input, pos_text, pos_mel)
            mel_input = t.cat([mel_input, postnet_pred[:,-1:,:]], dim=1)

        mag_pred = m_post.forward(postnet_pred)
        
    wav = spectrogram2wav(mag_pred.squeeze(0).cpu().numpy())
    write(hp.sample_path + "/synthesis.wav", hp.sr, wav)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step1', type=int, help='Global step to restore checkpoint for transformer', default=1594000)
    parser.add_argument('--restore_step2', type=int, help='Global step to restore checkpoint for postnet', default=774000)
    parser.add_argument('--max_len', type=int, help='Global step to restore checkpoint', default=400)

    args = parser.parse_args()
    unseen_sentence_01 = "Scientists at the CERN laboratory say they have discovered a new particle."
    unseen_sentence_02 = "There\'s a way to measure the acute emotional intelligence that has never gone out of style."
    unseen_sentence_03 = "President Trump met with other leaders at the Group of 20 conference."
    unseen_sentence_04 = "The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled."
    unseen_sentence_05 = "Generative adversarial network or variational auto-encoder."
    unseen_sentence_06 = "The buses aren't the problem, they actually provide a solution."
    unseen_sentence_07 = "You can call me directly at four two five seven zero three seven three four four or my cell four two five four four four seven four seven four or send me a meeting request with all the appropriate information."
    unseen_sentence_08 = "To deliver interfaces that are significantly better suited to create and process RFC eight twenty one , RFC eight twenty two , RFC nine seventy seven , and MIME content."
    unseen_sentence_09 = "Http0XX, Http1XX, Http2XX, Http3XX"
    unseen_sentence_10 = "was executed on a gibbet in front of his victim\'s house."
    unseen_sentence_11 = "For a while the preacher addresses himself to the congregation at large, who listen attentively"    
    unseen_sentence_12 = "Of course, once I became a full time musician, I discovered that many of those hard working, dedicated professionals also happened to be miscreant winos."
    unseen_sentence_13 = "Frozen, thawed cherries can be used in pies or be put in smoothies , but cherries can also be used in savory dishes."
    unseen_sentence_14 = "Symptoms may occur when the aneurysm grows or disrupts the wall of the aorta."
    unseen_sentence_15 = "When the sunlight strikes raindrops in the air they act as a prism and form a rainbow."
    unseen_sentence_16 = "The broader military presence was not meant to provoke conflict with Chinese."
    unseen_sentence_17 = "A rocket from Space X interacts with the individual beneath the soft flaw."
    unseen_sentence_18 = "Some have accepted it as a miracle without physical explanation."
    unseen_sentence_19 = "Six spoons of fresh snow peas five thick slabs of blue cheese and maybe a snack for her brother Bob."
    unseen_sentence_20 = "We also need a small plastic snake and a big toy frog for the kids."
    unseen_sentence_21 = "Throughout the centuries people have explained the rainbow in various ways."
    
    unseen_sentence_22 = "On offering to help the blind man, the man who then stole his car, had not, at that precise moment, had any evil intention, quite the contrary, what he did was nothing more than obey those feelings of generosity and altruism which, as everyone knows, are the two best traits of human nature and to be found in much more hardened criminals than this one, a simple car-thief without any hope of advancing in his profession, exploited by the real owners this enterprise, for it is they who take advantage of the needs of the poor."
        
    seen_sentence_01 = "the forms of printed letters should be beautiful, and that their arrangement on the page should be reasonable and a help to the shapeliness of the letters themselves."
    seen_sentence_02 = "especially as no more time is occupied or cost incurred in casting setting or printing beautiful letters"
    seen_sentence_03 = "the first books were printed in black letter ie the letter which was a gothic development of the ancient roman character"
    seen_sentence_04 = "I will quote an extract from the reverend gentleman’s own journal."
    seen_sentence_05 = "The result of the recommendation of the committee of 1862 was the Prison Act of 1865"
    seen_sentence_06 = "The felons’ side has a similar accommodation, and this mode of introducing the beverage is adopted because no publican as such"
    seen_sentence_07 = "He had prospered in early life, was a slop-seller on a large scale at Bury St. Edmunds, and a sugar-baker in the metropolis"
    seen_sentence_08 = "It has used other Treasury law enforcement agents on special experiments in building and route surveys in places to which the President frequently travels."
    seen_sentence_09 = "By this time the neighbors were aroused, and several people came to the scene of the affray."
    seen_sentence_10 = "Three years after the advent of the prison commissioners, it was decided that Newgate was an excessively costly and redundant establishment."
    seen_sentence_11 = "He has, however, doubly earned his sentence, and is actually condemned for burglary committed since his arrival in England."
           
    synthesis(seen_sentence_01, args)
