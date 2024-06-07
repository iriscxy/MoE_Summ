import json
import logging
import os
import pdb
import re
import time
from bert_score import score
import nltk
import numpy as np
from datasets import load_metric

logger = logging.getLogger(__name__)


class MetricCompute:
    rouge_metric = load_metric('metrics/rouge.py')

    def __init__(self, data_args, tokenizer, test_dataset, eval_datatset):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.eval_dataset = eval_datatset
        self.trainer = None

    def postprocess_text(self, metric_name, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        if self.data_args.chinese_data:
            if metric_name == 'rouge':  # 如果是中文数据且算rouge的话需要转数字
                split_char = lambda x: ' '.join([str(i) for i in self.tokenizer.encode(x, add_special_tokens=False)])
            else:  # 仅是中文数据，不是rouge指标，就只需要按字切分就行
                split_char = lambda x: ' '.join(list(x))
        else:
            split_char = lambda x: x.lower()

        # rougeLSum expects newline after each sentence
        if metric_name == "rouge":
            preds = ["\n".join([split_char(s) for s in nltk.sent_tokenize(pred)]) for pred in preds]
            labels = ["\n".join([split_char(s) for s in nltk.sent_tokenize(label)]) for label in labels]

        return preds, labels

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 获取当前状态，如果是evaluation就用eval数据集，如果是predict就用test数据集
        import traceback
        method_name = [s.name for s in traceback.extract_stack() if s.filename.endswith('trainer_seq2seq.py')]
        if len(method_name) == 0:
            logger.fatal(f'method name is none {method_name}')
        method_name = method_name[0]
        if method_name == 'predict':
            dataset = self.test_dataset
        else:
            dataset = self.eval_dataset

        # 准备展示文件，dec、ref、show（平行输入输出文件）
        # addi_source_str = tokenizer.batch_decode(dataset['addi_source'], skip_special_tokens=True)
        replace_special_token = lambda x: re.sub('\[.*?\]', '', x).replace('\n', ' ')
        if self.data_args.chinese_data:
            decoded_preds = [replace_special_token(p.replace(' ', '').strip()) for p in decoded_preds]
            decoded_labels = [replace_special_token(p.replace(' ', '').strip()) for p in decoded_labels]
        else:
            decoded_preds = [replace_special_token(p.strip()) for p in decoded_preds]
            decoded_labels = [replace_special_token(p.strip()) for p in decoded_labels]

        decoded_labels=[each.replace('\n',' ') for each in decoded_labels]
        decoded_preds=[each.replace('\n',' ') for each in decoded_preds]
        # decode_dir = os.path.join(self.data_args.log_root, f'decode-{self.trainer.state.global_step}')
        data_name = self.data_args.save_dataset_path.split('/')[-1]
        decode_dir = os.path.join(self.data_args.log_root + '/' + data_name, f'decode-{self.trainer.state.global_step}')

        if not os.path.exists(decode_dir):
            os.makedirs(decode_dir)
        fo_ref = open(os.path.join(decode_dir, 'reference.txt'), 'w', encoding='utf8')
        fo_dec = open(os.path.join(decode_dir, 'decoded.txt'), 'w', encoding='utf8')
        fo_show = open(os.path.join(decode_dir, 'show.txt'), 'w', encoding='utf8')
        input_content = self.tokenizer.batch_decode(dataset['input_ids'], skip_special_tokens=True)

        for pred, lab, inp_str in zip(decoded_preds, decoded_labels, input_content):  # , addi_source_str):
            fo_ref.write(f'{lab}\n')
            fo_dec.write(f'{pred}\n')
            if self.data_args.chinese_data:
                fo_show.write(f'{inp_str.replace(" ", "")}\n{lab}\n{pred}\n{"-" * 20}\n')
            else:
                fo_show.write(f'{inp_str}\n{lab}\n{pred}\n{"-" * 20}\n')
        result = {}
        # evaluate rouge
        P, R, F1 =score(decoded_preds, decoded_labels, lang='en', verbose=True)
        F1=F1.mean().item()
        rouge_decoded_preds, rouge_decoded_labels = self.postprocess_text('rouge', decoded_preds, decoded_labels)
        rouge_result = self.rouge_metric.compute(predictions=rouge_decoded_preds, references=rouge_decoded_labels,
                                                 use_stemmer=True)
        # Extract a few results from ROUGE
        result.update({key: value.mid.fmeasure * 100 for key, value in rouge_result.items()})

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        result['global_step'] = self.trainer.state.global_step
        result['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        result['bertscore']=F1
        data_name=self.data_args.save_dataset_path.split('/')[-1]
        result['data_name'] =data_name
        result['data_num'] =self.data_args.max_val_samples
        result['val_max_target_length'] =self.data_args.val_max_target_length

        fo_score = open(os.path.join(self.data_args.log_root, 'scores.txt'), 'a+', encoding='utf8')
        fo_score.write(f'{json.dumps(result)}\n')
        fo_score.close()
        return result
