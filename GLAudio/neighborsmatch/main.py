from argparse import ArgumentParser
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

from experiment import Experiment
from common import Task, STOP

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--task", dest="task", default=Task.NEIGHBORS_MATCH, type=Task.from_string, choices=list(Task),
                        required=False)
    parser.add_argument("--dim", dest="dim", default=32, type=int, required=False)
    parser.add_argument("--depth", dest="depth", default=2, type=int, required=False)
    parser.add_argument("--num_layers", dest="num_layers", default=2, type=int, required=False)
    parser.add_argument("--train_fraction", dest="train_fraction", default=0.8, type=float, required=False)
    parser.add_argument("--max_epochs", dest="max_epochs", default=50000, type=int, required=False)
    parser.add_argument("--eval_every", dest="eval_every", default=100, type=int, required=False)
    parser.add_argument("--batch_size", dest="batch_size", default=1024, type=int, required=False)
    parser.add_argument("--accum_grad", dest="accum_grad", default=1, type=int, required=False)
    parser.add_argument("--stop", dest="stop", default=STOP.TRAIN, type=STOP.from_string, choices=list(STOP),
                        required=False)
    parser.add_argument("--patience", dest="patience", default=20, type=int, required=False)
    parser.add_argument("--loader_workers", dest="loader_workers", default=0, type=int, required=False)
    parser.add_argument("--h", dest="h", default=0.05, type=float, required=False)
    parser.add_argument("--N", dest="N", default=80, type=int, required=False)

    args = parser.parse_args()
    Experiment(args).run()