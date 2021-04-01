"""
Entry point for training and evaluating the model.
"""

import argparse
import random
import logging
import time
import os
import tensorflow as tf

from data import read_dataset, split_dataset, augment_dataset
from model import OmniglotModel
from reptile import Reptile, FOML
from variables import weight_decay

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', help='evaluate a pre-trained model',
                        action='store_true', default=False)
    parser.add_argument('--checkpoint', help='checkpoint directory', default='model_checkpoint')
    parser.add_argument('--data-dir', help='data directory', default='data/omniglot', type=str)
    parser.add_argument('--seed', help='random seed', default=0, type=int)
    parser.add_argument('--classes', help='number of classes per inner task', default=5, type=int)
    parser.add_argument('--shots', help='number of examples per class', default=5, type=int)
    parser.add_argument('--train-shots', help='shots in a training batch', default=0, type=int)
    parser.add_argument('--inner-batch', help='inner batch size', default=5, type=int)
    parser.add_argument('--inner-iters', help='inner iterations', default=20, type=int)
    parser.add_argument('--replacement', help='sample with replacement', action='store_true')
    parser.add_argument('--learning-rate', help='Adam step size', default=1e-3, type=float)
    parser.add_argument('--meta-step', help='meta-training step size', default=0.1, type=float)
    parser.add_argument('--meta-step-final', help='meta-training step size by the end',
                        default=0.1, type=float)
    parser.add_argument('--meta-batch', help='meta-training batch size', default=1, type=int)
    parser.add_argument('--meta-iters', help='meta-training iterations', default=400000, type=int)
    parser.add_argument('--eval-batch', help='eval inner batch size', default=5, type=int)
    parser.add_argument('--eval-iters', help='eval inner iterations', default=50, type=int)
    parser.add_argument('--eval-samples', help='evaluation samples', default=10000, type=int)
    parser.add_argument('--eval-interval', help='train steps per eval', default=10, type=int)
    parser.add_argument('--weight-decay', help='weight decay rate', default=1, type=float)
    parser.add_argument('--transductive', help='evaluate all samples at once', action='store_true')
    parser.add_argument('--optimizer', help='optimization algorithm', default='adam', type=str)
    parser.add_argument('--foml', help='use FOML instead of Reptile', action='store_true')
    parser.add_argument('--foml-tail', help='number of shots for the final mini-batch in FOML',
                        default=None, type=int)
    parser.add_argument('--sgd', help='use vanilla SGD instead of Adam', action='store_true')
    args = parser.parse_args(args=args)

    return args

def train_kwargs(parsed_args):
    """
    Build kwargs for the train() function from the parsed
    command-line arguments.
    """
    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'train_shots': (parsed_args.train_shots or None),
        'inner_batch_size': parsed_args.inner_batch,
        'inner_iters': parsed_args.inner_iters,
        'replacement': parsed_args.replacement,
        'meta_step_size': parsed_args.meta_step,
        'meta_step_size_final': parsed_args.meta_step_final,
        'meta_batch_size': parsed_args.meta_batch,
        'meta_iters': parsed_args.meta_iters,
        'eval_inner_batch_size': parsed_args.eval_batch,
        'eval_inner_iters': parsed_args.eval_iters,
        'eval_interval': parsed_args.eval_interval,
        'weight_decay_rate': parsed_args.weight_decay,
        'transductive': parsed_args.transductive,
        'reptile_fn': _args_reptile(parsed_args)
    }

def evaluate_kwargs(parsed_args):
    """
    Build kwargs for the evaluate() function from the
    parsed command-line arguments.
    """
    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'eval_inner_batch_size': parsed_args.eval_batch,
        'eval_inner_iters': parsed_args.eval_iters,
        'replacement': parsed_args.replacement,
        'weight_decay_rate': parsed_args.weight_decay,
        'num_samples': parsed_args.eval_samples,
        'transductive': parsed_args.transductive,
        'reptile_fn': _args_reptile(parsed_args)
    }

def _args_reptile(parsed_args):
    if parsed_args.foml:
        return partial(FOML, tail_shots=parsed_args.foml_tail)
    return Reptile

# pylint: disable=R0913,R0914
def train(sess,
          model,
          train_set,
          test_set,
          save_dir,
          num_classes=5,
          num_shots=5,
          inner_batch_size=5,
          inner_iters=20,
          replacement=False,
          meta_step_size=0.1,
          meta_step_size_final=0.1,
          meta_batch_size=1,
          meta_iters=400000,
          eval_inner_batch_size=5,
          eval_inner_iters=50,
          eval_interval=10,
          weight_decay_rate=1,
          time_deadline=None,
          train_shots=None,
          transductive=False,
          reptile_fn=Reptile,
          log_fn=print):
    """
    Train a model on a dataset.
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    saver = tf.train.Saver()
    reptile = reptile_fn(sess,
                         transductive=transductive,
                         pre_step_op=weight_decay(weight_decay_rate))
    accuracy_ph = tf.placeholder(tf.float32, shape=())
    tf.summary.scalar('accuracy', accuracy_ph)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(save_dir, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(save_dir, 'test'), sess.graph)
    tf.global_variables_initializer().run()
    sess.run(tf.global_variables_initializer())
    for i in range(meta_iters):
        frac_done = i / meta_iters
        cur_meta_step_size = frac_done * meta_step_size_final + (1 - frac_done) * meta_step_size
        reptile.train_step(train_set, model.input_ph, model.label_ph, model.minimize_op,
                           num_classes=num_classes, num_shots=(train_shots or num_shots),
                           inner_batch_size=inner_batch_size, inner_iters=inner_iters,
                           replacement=replacement,
                           meta_step_size=cur_meta_step_size, meta_batch_size=meta_batch_size)
        if i % eval_interval == 0:
            accuracies = []
            for dataset, writer in [(train_set, train_writer), (test_set, test_writer)]:
                correct = reptile.evaluate(dataset, model.input_ph, model.label_ph,
                                           model.minimize_op, model.predictions,
                                           num_classes=num_classes, num_shots=num_shots,
                                           inner_batch_size=eval_inner_batch_size,
                                           inner_iters=eval_inner_iters, replacement=replacement)
                summary = sess.run(merged, feed_dict={accuracy_ph: correct/num_classes})
                writer.add_summary(summary, i)
                writer.flush()
                accuracies.append(correct / num_classes)
            log_fn('batch %d: train=%f test=%f' % (i, accuracies[0], accuracies[1]))
        if i % 100 == 0 or i == meta_iters-1:
            saver.save(sess, os.path.join(save_dir, 'model.ckpt'), global_step=i)
        if time_deadline is not None and time.time() > time_deadline:
            break

# pylint: disable=R0913,R0914
def evaluate(sess,
             model,
             dataset,
             num_classes=5,
             num_shots=5,
             eval_inner_batch_size=5,
             eval_inner_iters=50,
             replacement=False,
             num_samples=10000,
             transductive=False,
             weight_decay_rate=1,
             reptile_fn=Reptile):
    """
    Evaluate a model on a dataset.
    """
    reptile = reptile_fn(sess,
                         transductive=transductive,
                         pre_step_op=weight_decay(weight_decay_rate))
    total_correct = 0
    for _ in range(num_samples):
        total_correct += reptile.evaluate(dataset, model.input_ph, model.label_ph,
                                          model.minimize_op, model.predictions,
                                          num_classes=num_classes, num_shots=num_shots,
                                          inner_batch_size=eval_inner_batch_size,
                                          inner_iters=eval_inner_iters, replacement=replacement)
    return total_correct / (num_samples * num_classes)


def main(args=None):
    args = parse_args(args=args)
    print(args)

    random.seed(args.seed)

    train_set, test_set = split_dataset(read_dataset(args.data_dir))

    logging.info(f"Loaded {len(train_set)} training examples.")
    logging.info(f"Loaded {len(test_set)} test examples.")
    
    train_set = list(augment_dataset(train_set))
    test_set = list(test_set)

    model = OmniglotModel(args.classes, args.optimizer, args.learning_rate)

    with tf.Session() as sess:
        if not args.pretrained:
            train(sess, model, train_set, test_set, args.checkpoint, **train_kwargs(args))
        else:
            print('Restoring from checkpoint...')
            tf.train.Saver().restore(sess, tf.train.latest_checkpoint(args.checkpoint))


    eval_kwargs = evaluate_kwargs(args)
    print('Train accuracy: ' + str(evaluate(sess, model, train_set, **eval_kwargs)))
    print('Test accuracy: ' + str(evaluate(sess, model, test_set, **eval_kwargs)))
    

if __name__ == '__main__':
    main()