import os
import logging
import torch
import numpy as np
import random
from conf.config import get_data_path, args_config
from datautil.dataloader import load_dataset
from vocab.dep_vocab import create_vocab
from modules.model import ParserModel
from pathlib import Path
import jieba
import time
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def set_seeds(seed=1349):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(input_path, result_path):
    set_seeds(3347)
    print('cuda available:', torch.cuda.is_available())
    print('cuDnn available:', torch.backends.cudnn.enabled)
    print('GPU numbers:', torch.cuda.device_count())
    data_path = get_data_path("./conf/datapath.json")
    char_vocab, bichar_vocab = create_vocab(data_path['data']['test_data'])
    char_embed_weights = char_vocab.get_embedding_weights(data_path['pretrained']['char_embedding'])
    bichar_embed_weights = bichar_vocab.get_embedding_weights(data_path['pretrained']['bichar_embedding'])

    args = args_config()
    args.char_vocab_size = char_vocab.vocab_size
    args.bichar_vocab_size = bichar_vocab.vocab_size
    args.tag_size = char_vocab.tag_size
    args.rel_size = char_vocab.rel_size

    parser_model = ParserModel(args, char_embed_weights, bichar_embed_weights)
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
        # if torch.cuda.device_count() > 1:
        #     parser_model = nn.DataParallel(parser_model, device_ids=list(range(torch.cuda.device_count() // 2)))
    else:
        args.device = torch.device('cpu')

    parser_model = parser_model.to(args.device)
    print('模型参数量：', sum(p.numel() for p in parser_model.parameters() if p.requires_grad))

    biff_parser = torch.load('./model/model.pkl')
    json_obj = []
    for file in Path(input_path).glob('**/*.txt'):
        logging.info("file: {}".format(file))
        mid = os.path.relpath(str(file), input_path)
        logging.info("mid: {}".format(mid))
        dst_json = os.path.join(result_path, os.path.dirname(mid), str(file.stem) + '.json')
        logging.info("dst_json: {}".format(dst_json))
        os.makedirs(os.path.dirname(dst_json), exist_ok=True)

        sentences = []
        with open("./test.txt", 'w') as wf:
            with open(str(file), 'r') as f:
                for line in f:
                    tokens = [word for word in jieba.cut(line.replace("\n", ""))]
                    sentences.append(tokens)
                    for idx, token in enumerate(tokens):
                        wf.write(str(idx) + "\t" + token + "\t_\t_\t_\t_\t_\tamod\t_\t_\n")
                    wf.write("\n")

        test_data = load_dataset("./test.txt", char_vocab)
        print('test data size:', len(test_data))
        os.remove("./test.txt")
        test_start_time = time.time()
        pred_seg_all_list = biff_parser.test(test_data, args, char_vocab, bichar_vocab)
        test_time = time.time() - test_start_time

        for idx, pred_tokens in enumerate(pred_seg_all_list):
            sentence_json_obj = {}
            sentence_json_obj["ID"] = idx
            sentence_json_obj["text"] = "".join(sentences[idx])
            sentence_json_obj["words"] = []
            for id, item in enumerate(pred_tokens):
                if id == 0:
                    continue
                token_obj = {}
                token_obj["id"] = id
                token_obj["form"] = sentences[idx][id - 1]
                token_obj["head"] = item.get_head()
                token_obj["pos"] = item.get_tag()
                token_obj["deprel"] = item.get_deprel()
                token_obj["stanfordnlpdependencies"] = ""
                sentence_json_obj["words"].append(token_obj)
            json_obj.append(sentence_json_obj)
        with open(dst_json, 'w', encoding='utf-8') as f:
            json.dump(json_obj, f, indent=4, ensure_ascii=False, cls=NpEncoder)
        return len(json_obj), test_time, json_obj