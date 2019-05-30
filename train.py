import gpt_2_simple as gpt2
import argparse

parser = argparse.ArgumentParser(
    description='Fine-tune GPT-2 on your custom dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', metavar='PATH', type=str, required=True, help='Input file, directory, or glob pattern (utf-8 text, or preencoded .npz files).')
parser.add_argument('--model_name', metavar='MODEL', type=str, default='345M', help='Pretrained model name')

def main:
    args = parser.parse_args()

    model_name = args.model_name
    data_set = args.dataset
    gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/117M/

    sess = gpt2.start_tf_sess()
    gpt2.finetune(sess,
        data_set,
        model_name=model_name,
        steps=1000)   # steps is max number of training steps

    gpt2.generate(sess)

if __name__ == '__main__':
    main()
