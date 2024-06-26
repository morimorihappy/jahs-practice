from typing import Any
import optuna
import jahs_bench
import pandas as pd
import warnings

# 警告を抑制
warnings.filterwarnings("ignore")


class ObjectiveFunc:
    def __init__(self, dataset_name: str) -> None:
        self.benchmark = jahs_bench.Benchmark(task=dataset_name, download=True)

    def trial_to_config(self, trial: optuna.trial.Trial) -> dict:
        # JAHS-Benchのcifar10タスクに必要なハイパーパラメータ
        config = {
            "LearningRate": trial.suggest_float("LearningRate", 1e-3, 1e0, log=True),
            "BatchSize": trial.suggest_categorical("BatchSize", [16, 32, 64, 128]),
            "Optimizer": trial.suggest_categorical("Optimizer", ["SGD"]),
            "Momentum": trial.suggest_float("Momentum", 0.0, 1.0) if trial.params.get("Optimizer") == "SGD" else 0.0,
            "Activation": trial.suggest_categorical("Activation", ["ReLU", "Hardswish", "Mish"]),
            "N": trial.suggest_int("N", 1, 5),
            "Op1": trial.suggest_categorical("Op1", list(range(5))),
            "Op2": trial.suggest_categorical("Op2", list(range(5))),
            "Op3": trial.suggest_categorical("Op3", list(range(5))),
            "Op4": trial.suggest_categorical("Op4", list(range(5))),
            "Op5": trial.suggest_categorical("Op5", list(range(5))),
            "Op6": trial.suggest_categorical("Op6", list(range(5))),
            "Resolution": trial.suggest_categorical("Resolution", [0.25, 0.5, 1.0]),
            "TrivialAugment": trial.suggest_categorical("TrivialAugment", [True, False]),
            "W": trial.suggest_int("W", 8, 64),
            "WeightDecay": trial.suggest_float("WeightDecay", 1e-5, 1e-2, log=True)
        }
        return config
    
    def __call__(self, trial: optuna.trial.Trial) -> float:
        config = self.trial_to_config(trial)
        results = self.benchmark(config, nepochs=200)
        return 100 - results[200]["valid-acc"]

# Optunaの最適化を実行
func = ObjectiveFunc(dataset_name="cifar10")
study = optuna.create_study(direction="minimize")
study.optimize(func, n_trials=100)

# 結果を表示
print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
