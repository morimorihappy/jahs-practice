import random

# Sphere関数の定義
def sphere(x, y):
    return x**2 + y**2

# 探索範囲の設定
x_min, x_max = -5.0, 5.0
y_min, y_max = -5.0, 5.0

# ランダムサーチの設定
n_trials = 1000  # 試行回数

# 最良の結果の初期化
best_x = None
best_y = None
best_value = float('inf')

# ランダムサーチの実行
for _ in range(n_trials):
    x = random.uniform(x_min, x_max)
    y = random.uniform(y_min, y_max)
    value = sphere(x, y)
    
    if value < best_value:
        best_value = value
        best_x = x
        best_y = y

# 結果の表示
print(f"Best value: {best_value}")
print(f"Best coordinates: x = {best_x}, y = {best_y}")
