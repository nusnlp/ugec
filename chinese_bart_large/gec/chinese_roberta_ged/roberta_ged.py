import argparse
from fairseq.data.data_utils import collate_tokens
from fairseq.models.chinese_roberta import ChineseRobertaModel

def main(args):
    roberta = ChineseRobertaModel.from_pretrained(
        args.weight_path,
        checkpoint_file=args.checkpoint,
        data_name_or_path='data-bin'
    )

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )
    ncorrect, nsamples = 0, 0
    roberta.cuda()
    roberta.eval()
    fin = open(args.input_data).readlines()
    skip_path = args.output_file + ".skip_idx"
    with open(args.output_file, "w+", encoding="utf-8") as out_f, open(skip_path, "w+", encoding="utf-8") as skip_f:
        batch_list = []
        for index, line in enumerate(fin):
            # print(index)
            tokens = line.strip()
            if len(batch_list) == 64:
                batch = collate_tokens([roberta.encode(pair[0]) for pair in batch_list], pad_idx=1)
                logits = roberta.predict('sentence_classification_head', batch)
                predictions = logits.argmax(dim=1)
                for idx, prediction in enumerate(predictions):
                    prediction_label = label_fn(prediction.item())
                    out_f.write(batch_list[idx][0] + "\t" + str(prediction_label) + "\n")
                batch_list = []
                batch_list.append(tokens)
            else:
                if len(roberta.encode(tokens)) <= 512:
                    batch_list.append([tokens])
                else:
                    skip_f.write(str(index) + "\n")


        if batch_list:
            batch = collate_tokens([roberta.encode(pair[0]) for pair in batch_list], pad_idx=1)
            logits = roberta.predict('sentence_classification_head', batch)
            predictions = logits.argmax(dim=1)
            for idx, prediction in enumerate(predictions):
                prediction_label = label_fn(prediction.item())
                out_f.write(batch_list[idx][0] + "\t" + str(prediction_label) + "\n")


if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path',
                        type=str,
                        help='path to the checkpoint folder')
    parser.add_argument('--checkpoint',
                        type=str,
                        help='checkpoint name')
    parser.add_argument('--input_data',
                        type=str,
                        help='input prediction file')
    parser.add_argument('--output_file',
                        type=str,
                        help='path to the output result')
    args = parser.parse_args()
    main(args)
