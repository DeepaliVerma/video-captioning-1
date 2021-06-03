import torch
import numpy as np
from keras.preprocessing.text import Tokenizer
import json
from config import myConfig
import os
from model.Seq2Seq import Encoder,Decoder
from model.local_constructor import *
from collections import OrderedDict
import time
import argparse
from extra_features.extract_features import extractVideoFrames,i3dFeatures,loadModel

class VideoCaptionGenerate(object):
    def __init__(self,config,json):
        self.config=config
        self.numDecoderTokens=config.numDecoderTokens
        self.vocaburayList=[]
        self.contentList=[]
        self.JsonPath=json
        self.JsonContent=None

        self.inputSentenceLength=config.inputSentenceLength

        self.Tokenizer=Tokenizer(self.numDecoderTokens)
        self.getTokenizer()

        self.featuresPath=config.featuresPath
        self.features = self.loadFeatures()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder=None
        self.decoder=None
        self.load_model()

    def getTokenizer(self):
        with open(self.JsonPath, encoding='utf-8-sig', errors='ignore') as file:
            self.JsonContent = json.load(file, strict=False)
            for video in self.JsonContent.keys():
                content=self.JsonContent[video]
                for caption in content['caption']:
                    caption = "<bos> " + caption + " <eos>"
                    if len(caption.split()) > self.config.longestSentence or len(caption.split()) < self.config.shortestSentence:
                        continue
                    else:
                        self.contentList.append([caption,video])

        for content in self.contentList:
            self.vocaburayList.append(content[0])

        self.Tokenizer.fit_on_texts(self.vocaburayList)

    def loadFeatures(self):
        featureDict={}
        featureFiles=os.listdir(self.featuresPath)
        for featureFile in featureFiles:
            videoName=featureFile.split(".")[0]
            featureDict[videoName]=os.path.join(self.featuresPath,featureFile)
        return featureDict

    def load_model(self):
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)

        checkpoint = torch.load(self.config.checkpointPath)
        net_parameters = checkpoint["net"]
        encoder_parameters = OrderedDict()
        decoder_parameters = OrderedDict()

        for key in net_parameters.keys():
            if "encoder" in key:
                new_key = key.replace("encoder.e", "e")
                encoder_parameters[new_key] = net_parameters[key]
            if "decoder" in key:
                new_key = key.replace("decoder.d", "d")
                decoder_parameters[new_key] = net_parameters[key]

        self.encoder.load_state_dict(encoder_parameters)
        self.encoder.eval()
        self.encoder.to(self.device)

        self.decoder.load_state_dict(decoder_parameters)
        self.decoder.eval()
        self.decoder.to(self.device)

    def label2word(self):
        word = {value: key for key, value in self.Tokenizer.word_index.items()}
        return word

    def beam_search(self,config, beam_width, vocab, decoder, input, hidden, encoder_outputs):
        input_list = [input]
        hidden_list = [hidden]
        cum_prob_list = [torch.cuda.FloatTensor([1. for _ in range(config.batch_size)])]
        cum_prob_list = [torch.log(cum_prob) for cum_prob in cum_prob_list]

        output_list = [[[]] for _ in range(config.batch_size)]
        for t in range(config.caption_max_len + 1):

            outputs = None
            tmp_next_hidden_list = []
            for i, (input, hidden, cum_prob) in enumerate(zip(input_list, hidden_list, cum_prob_list)):
                output, next_hidden = decoder(input, hidden, encoder_outputs)
                tmp_next_hidden_list.append(next_hidden)

                np_output_list = np.asarray(output_list)
                EOS_row_idxs, EOS_col_idxs = np.where(np_output_list[:, i] == vocab.word2idx['<EOS>'])
                sequence_length = np.empty(config.batch_size)
                sequence_length.fill(t + 1)
                sequence_length[EOS_row_idxs] = EOS_col_idxs + 1
                sequence_length = sequence_length ** 0.7
                sequence_length = torch.cuda.FloatTensor(sequence_length)

                cum_prob = cum_prob / sequence_length
                cum_prob = cum_prob.unsqueeze(1).expand_as(output)
                output = torch.log(torch.sigmoid(output))
                output += cum_prob
                outputs = output if outputs is None else torch.cat((outputs, output), dim=1)
            topk_probs, topk_flat_idxs = outputs.topk(beam_width)

            topk_probs = topk_probs.transpose(0, 1)
            topk_flat_idxs = topk_flat_idxs.transpose(0, 1)
            topk_idxs = topk_flat_idxs % vocab.n_vocabs
            topk_is = topk_flat_idxs // vocab.n_vocabs

            next_input_list = topk_idxs.clone()

            next_cum_prob_list = topk_probs.clone()

            if config.decoder_model == "LSTM":
                next_hidden_list = []
                next_context_list = []
                for topk_idx, topk_i in zip(topk_idxs, topk_is):
                    next_hidden = []
                    next_context = []
                    for b, (i, k) in enumerate(zip(topk_idx, topk_i)):
                        next_hidden.append(tmp_next_hidden_list[k][0][:, b])
                        next_context.append(tmp_next_hidden_list[k][1][:, b])
                    next_hidden = torch.cat(next_hidden)
                    next_context = torch.cat(next_context)
                    next_hidden = next_hidden.unsqueeze(0)
                    next_context = next_context.unsqueeze(0)
                    next_hidden_list.append(next_hidden)
                    next_context_list.append(next_context)
                next_hidden_list = [(h, c) for h, c in zip(next_hidden_list, next_context_list)]
            else:
                next_hidden_list = []
                for topk_idx, topk_i in zip(topk_idxs, topk_is):
                    next_hidden = []
                    for b, (i, k) in enumerate(zip(topk_idx, topk_i)):
                        next_hidden.append(tmp_next_hidden_list[k][:, b])
                    next_hidden = torch.cat(next_hidden)
                    next_hidden = next_hidden.unsqueeze(0)
                    next_hidden_list.append(next_hidden)

            next_output_list = [[] for _ in range(config.batch_size)]
            for topk_idx, topk_i in zip(topk_idxs, topk_is):
                for b, (i, k) in enumerate(zip(topk_idx, topk_i)):
                    next_output_list[b].append(output_list[b][k] + [i.item()])

            input_list = [input.unsqueeze(0) for input in next_input_list]
            hidden_list = next_hidden_list
            cum_prob_list = next_cum_prob_list
            output_list = next_output_list

            if t == config.caption_max_len or torch.all(torch.cat(input_list) == 0):
                break

        top1_output_list = [output[0] for output in output_list]
        return top1_output_list

    def greedy_generate(self, features):
        self.encoder.eval()
        self.decoder.eval()
        features=features.to(self.device)
        word_map = self.label2word()
        _,states = self.encoder(features)
        target_seq = np.zeros((1, 1, 1500))
        final_sentence = ''
        target_seq[0, 0, self.Tokenizer.word_index['bos']] = 1
        target_seq=torch.from_numpy(target_seq)
        target_seq = target_seq.to(self.device,dtype=torch.float32)
        self.decoder.get_init_state(states)
        for i in range(10):
            outputs,states= self.decoder(target_seq)
            outputs = outputs.contiguous().view(-1, self.config.tokenizerOutputdims)
            _, predicted = outputs.max(1)
            predicted=predicted.cpu().item()
            if predicted == 0:
                continue
            if word_map[predicted] == 'eos':
                break
            final_sentence = final_sentence + word_map[predicted] + ' '
            target_seq = np.zeros((1, 1, 1500))
            target_seq[0, 0, predicted] = 1
            target_seq = torch.from_numpy(target_seq)
            target_seq = target_seq.to(self.device,dtype=torch.float32)
            self.decoder.get_init_state(states)
        return final_sentence

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--videoFeature', default='/mnt/MSR-VTT/MSR-VTT-i3dfeatures/video6122.npy', type=str,help='path to input video feature')
    parser.add_argument('--inputType', default='feature', type=str, help='/feature/ means input feature'
                                                                         '/video/ means input video,and extract the feature manually')
    args = parser.parse_args()

    load_start = time.time()
    captionModel=VideoCaptionGenerate(config=myConfig,json=myConfig.totalJson)
    load_end = time.time()
    print('==========================')
    print('It took {:.2f} seconds to init video-caption model'.format(load_end - load_start))

    if args.inputType=="feature":
        features = np.load(args.videoFeature)
        features = torch.from_numpy(features)
        features = features.view(1, -1, myConfig.inputEncoderDims)
    else:
        raise KeyError

    start = time.time()
    caption=captionModel.greedy_generate(features)
    end = time.time()
    sentence=""
    for slice in caption.split():
        sentence = sentence + ' ' + slice
        print('==========================')
        print(sentence)

    print('==========================')
    print('It took {:.2f} seconds to generate caption'.format(end - start))


