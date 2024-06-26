# jahs-practice

## jahs-practice.py
実行方法！！！

ベンチマーククラスのインスタンスを作成．
cifar10という画像分類タスク，必要なデータのダウンロードを指定．
```bash
benchmark = jahs_bench.Benchmark(task="cifar10", download=True)
```

ベンチマークからランダムなニューラルネットワークの設定（ハイパーパラメータのセット）を取得．
```bash
config = benchmark.sample_config()
```

取得した設定を使用してベンチマークを実行．トレーニングのエポック数は200に指定．
```bash
results = benchmark(config, nepochs=200)
```
