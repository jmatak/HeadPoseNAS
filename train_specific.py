import argparse

from data.dataset import *
from model.network import *
from model.representation import *
from training.train import *

tf.random.set_random_seed(1950)
random.seed(1950)
np.random.seed(1950)


def parse_model(name):
    ind_str = name.split("_")[1]
    ind = [int(i) for i in ind_str]
    return ind


def train(args, blocks, kernels, flops):
    if args.gpu:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
    else:
        session = tf.Session()
    layer = get_placeholder(args.model, args.layer)
    dataset_train, dataset_val, dataset_test = get_train_test_datasets(args.machine, args.model,
                                                                       args.train_set, args.val_set, args.test_set,
                                                                       layer["name"], args.batch_size, session)

    inpt, output, training = master_module(layer["shape"], blocks, kernels, flops)

    trainer = PoseTrainer(inpt, output, training, session, name=args.name)
    session.run(tf.global_variables_initializer())

    val_loss = trainer.train(dataset_train, dataset_val, dataset_test,
                             epochs=args.epochs,
                             learning_rate=args.learning_rate)

    open(f"{MODEL_SAVER_PATH}/{args.name}/{args.name}.txt", "w+").write(
        json.dumps({
            "NAME": args.name,
            "VAL_LOSS": trainer.validation_loss,
            "FLOPS": trainer.validation_loss,
            "FIT": trainer.fitness
        }, indent=3)
    )

    session.close()
    tf.reset_default_graph()
    return val_loss


def test(args, blocks, kernels, flops):
    if args.gpu:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
    else:
        session = tf.Session()
    layer = get_placeholder(args.model, args.layer)

    inpt, output, training = master_module(layer["shape"], blocks, kernels, flops)

    model_dir = f"{MODEL_SAVER_PATH}/{args.name}/{args.name}.ckpt"
    trainer = PoseTrainer(inpt, output, training, session, name=args.name, export_dir=model_dir)

    for test_set in args.test_sets:
        print(test_set)
        dataset_test = get_test_dataset(args.machine, args.model,
                                        test_set, layer["name"],
                                        args.batch_size, session)
        res = trainer.test_forward(dataset_test)
        print(res)
    session.close()
    tf.reset_default_graph()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose extractor trainer.')

    parser.add_argument('-ma', "--machine", type=str, default="ludmila")
    parser.add_argument('-d', "--train_set", type=str, default="300w_train")
    parser.add_argument('-v', "--val_set", type=str, default="300w_val")
    parser.add_argument('-te', "--test_set", type=str, default="afbi")
    parser.add_argument('-t', "--test_sets", nargs='+', default=["aflw", "biwi"])
    parser.add_argument('-m', "--model", type=str, default="inception")

    parser.add_argument('-la', "--layer", type=int, default=13)
    parser.add_argument('-lr', "--learning_rate", nargs='+',
                        default=[0.0005, 0.0001, 0.00005, 0.00004, 0.00002, 0.00009])
    parser.add_argument('-b', "--batch_size", type=int, default=32)
    parser.add_argument('-e', "--epochs", type=int, default=6)
    parser.add_argument('-g', "--gpu", type=bool, default=True)
    parser.add_argument("--test", type=bool, default=False)

    parser.add_argument('-n', "--name", type=str, default="baseline")
    parser.add_argument('-ind', "--individual", type=str, default=None)

    args = parser.parse_args()

    if args.individual:
        state = NeuralSearchState()
        args.name = f"{test}_{args.individual}"
        model = state.decode_int(parse_model(args.name))
    else:
        model = [ConvBlock,
                 ConvBlockUpscale,
                 ConvBlock,
                 ConvBlock,
                 ConvBlockUpscale,
                 ConvBlock
                 ], [1, 3, 3, 3, 3, 1], 2

    model_args = args, *model

    if not args.test:
        res = train(*model_args)
        test(*model_args)
    else:
        res = test(*model_args)

    print("Result {}".format(res))
