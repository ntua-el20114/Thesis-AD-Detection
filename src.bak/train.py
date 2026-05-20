import argparse
import torch

from model import utils

log = utils.get_logger()


def main(args):
    # load data
    log.debug(f"Loading data from {args.data}.")
    data = utils.load_pkl(args.data)
    log.info("Loaded data.")

    # trainset = Dataset(data["train"], args.batch_size)
    # devset = Dataset(data["dev"], args.batch_size)
    # testset = Dataset(data["test"], args.batch_size)
 
    log.debug("Building model...")
    # model_file = "./save/model.pt"
    # model = DialogueGCN(args).to(args.device)
    # opt = Optim(args.learning_rate, args.max_grad_value, args.weight_decay)
    # opt.set_parameters(model.parameters(), args.optimizer)

    # coach = Coach(trainset, devset, testset, model, opt, args)
    # if not args.from_begin:
    #     ckpt = torch.load(model_file)
    #     coach.load_ckpt(ckpt)

    # Train.
    log.info("Start training...")
    # ret = coach.train()

    # Save.
    # checkpoint = {
    #     "best_dev_f1": ret[0],
    #     "best_epoch": ret[1],
    #     "best_state": ret[2],
    # }
    # torch.save(checkpoint, model_file)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data")
    parser.add_argument("--seed", type=int, default=3559,
                        help="Random seed.")
    parser.add_argument("--reps", type=int, default=1,
                        help="Number of experiment repetitions.")

    # Training Parameters
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Computing device.")
    parser.add_argument("--epochs", default=1, type=int,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-8,
                        help="Weight decay.")
    parser.add_argument("--max_grad_value", default=-1, type=float,
                        help="""If the norm of the gradient vector exceeds this,
                        normalize it to have the norm equal to max_grad_norm""")
    parser.add_argument("--drop_rate", type=float, default=0.5,
                        help="Dropout rate.")

    # Model Parameters 
    parser.add_argument("--hidden_size", type=int, default=100,
                        help="Network Hidden Dimention")
    parser.add_argument("--co_att_heads", type=int, default=8,
                         help="Co-attention Heads")
    parser.add_argument("--co_att_layers", type=int, default=2,
                         help="Co-attention Layers")

    args = parser.parse_args()
    log.debug(args)

    main(args)
