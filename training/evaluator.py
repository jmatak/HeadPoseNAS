import os

from data.dataset import *
from model.model import master_module
from model.network import *
from training.train import *

tf.random.set_random_seed(1950)
random.seed(1950)
np.random.seed(1950)


def check_result(name):
    info = json.load(open(f"{MODEL_SAVER_PATH}/{name}/{name}.txt", "r"))
    return float(info["FIT"])


def evaluate(args, state, ind):
    name = "model_" + "".join(str(c) for c in state.repr_int(ind))
    if os.path.exists(f"{MODEL_SAVER_PATH}/{name}"):
        result = check_result(name)
        open(args.logger, "a").write(f"{name}:{result}\n")
        return result

    if args.gpu:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
    else:
        session = tf.Session()
    layer = get_placeholder(args.model, args.layer)
    dataset_train, dataset_val = get_train_datasets(args.machine, args.model,
                                                    args.train_set, args.val_set,
                                                    layer["name"], args.batch_size, session)

    blocks, kernels, flops = state.decode(ind)

    inpt, output, training = master_module(layer["shape"], blocks, kernels, flops)

    trainer = PoseTrainer(inpt, output, training, session, name=name)
    session.run(tf.global_variables_initializer())

    trainer.train(dataset_train, dataset_val,
                  epochs=args.epochs,
                  learning_rate=args.learning_rate)

    open(f"{MODEL_SAVER_PATH}/{name}/{name}.txt", "w+").write(
        json.dumps({
            "NAME": name,
            "VAL_LOSS": trainer.validation_loss,
            "FLOPS": trainer.flops,
            "FIT": trainer.fitness
        }, indent=3)
    )
    open(args.logger, "a").write(f"{name}:{trainer.fitness}\n")
    session.close()
    tf.reset_default_graph()
    return trainer.fitness


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
