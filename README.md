# optimization-foundations
Learning deep learning and optimization starting from D2L.
Study Notes: Deep Learning Basics (D2L & PyTorch Autograd)このノートは、『Dive into Deep Learning (D2L)』の学習を通じた環境構築のトラブルシューティングと、PyTorchにおける自動微分の基礎概念をまとめたものです。1. 環境構築のトラブルシューティングPython 3.12+ における NumPy インストールエラー現象: AttributeError: module 'pkgutil' has no attribute 'ImpImporter' が発生し、d2l ライブラリのインストールが止まる。原因: Python 3.12以降で廃止された ImpImporter を、古い NumPy (1.23.5) が参照しているため。解決策:先に最新の NumPy を入れる: pip install "numpy>=1.26.0"依存関係を無視して d2l を入れる: pip install d2l==1.0.3 --no-deps不足しているライブラリを手動追加: pip install matplotlib requests pandas2. 数値微分の基礎微分の定義に基づき、変化量 $h$ を限りなく $0$ に近づけた際の傾きの収束を確認。Pythonimport numpy as np

# 微分係数の近似
for h in 10.0**np.arange(-1, -6, -1):
    print(f'h={h:.5f}, numerical limit={(f(1+h)-f(1))/h:.5f}')
3. PyTorch 自動微分 (Autograd) の核心PyTorchにおけるバックプロパゲーション（誤差逆伝播法）の重要なメソッドと挙動のまとめ。基本的な流れ勾配の保存準備: x.requires_grad_(True) で計算履歴の記録を開始。順伝播 (Forward): $y = f(x)$ を計算。逆伝播 (Backward): y.backward() で出力から入力へ向かって微分を実行。勾配の取得: x.grad に結果 $\frac{dy}{dx}$ が格納される。重要なキーワードメソッド / 属性役割.backward()逆伝播のトリガー。連鎖律（Chain Rule）を用いて勾配を計算する。.grad計算された勾配（微分の値）が格納されるバッファ。.grad.zero_()勾配のリセット。PyTorchは勾配を累計（加算）するため、新しい計算の前に必須。.detach()計算グラフから切り離す。値はそのままで、履歴を消去して定数として扱う。4. 応用的な挙動非スカラー変数のバックワード出力がスカラー（1つの数値）でない場合、PyTorchはそのままでは backward() できません。解決策1: y.sum().backward() で合計してスカラーにする。解決策2: y.backward(torch.ones(len(y))) のように、各要素の重みを指定する。制御フローを伴う自動微分PyTorchはPythonの if 文や while 文などの動的な制御フローを通した計算でも、正しく微分を計算できます。Pythondef f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
# 検証: f(a) は常に a * k (kは定数) の形になるため、d/a と勾配は一致する
print(a.grad == d / a) 
5. 学習のポイント逆向きに微分する理由: 出力（誤差）から遡ることで、数百万のパラメータに対する勾配をたった1回のパスで効率よく計算できるため。計算グラフの切断 (detach): 強化学習やGANなど、特定の変数を更新したくない（定数として扱いたい）場合に非常に強力なツールとなる。

# d2l 学習ノート：2.6 確率 (Probability)
### ―― 機械学習における「不確実性」の扱い ――

## 1. 確率論の基礎（形式的な定義）
機械学習における推論を支える数学的枠組み。

* **標本空間 ($\Omega$):** 起こりうるすべての結果の集合。
    * 例：コイン投げ $\{H, T\}$、サイコロ $\{1, 2, 3, 4, 5, 6\}$
* **事象 (Event):** 標本空間の部分集合。
* **コルモゴロフの公理 (Kolmogorov's Axioms):**
    1. どの事象 $A$ についても $P(A) \ge 0$
    2. 全事象の確率は 1 ($P(\Omega) = 1$)
    3. 互いに排反な事象の和集合の確率は、個別の確率の和になる。

[Image of Venn diagram showing disjoint events and their union within sample space Omega]

## 2. 確率変数 (Random Variables)
結果に数値を割り当てる「写像」。
* **離散型:** 飛び飛びの値（例：ポケモンの個体値、命中・急所判定）。
* **連続型:** 連続した値（例：体重、ダメージ計算の乱数倍率）。
    * 連続型は「1点」の確率は 0 になるため、**確率密度関数 (PDF)** の積分で確率を計算する。

## 3. 結合確率・条件付き確率・ベイズ則
複数の変数が絡む「推論」の核心。

* **結合（同時）確率:** $P(X, Y) = P(Y | X) P(X)$
* **周辺化:** 他の変数の影響を足し合わせて（積分して）消し、特定の変数の分布を出す。
* **ベイズの定理:**
  $$P(A | B) = \frac{P(B | A) P(A)}{P(B)}$$
    * **事前確率 $P(A)$:** データを観測する前の確信度。
    * **尤度 (ゆうど) $P(B | A)$:** 仮説 $A$ のもとでデータ $B$ が得られる確率。
    * **事後確率 $P(A | B)$:** 新しいデータを得て更新された確信度。

[Image of Bayesian updating process showing how prior distribution and likelihood result in a posterior distribution]

## 4. HIV検査の例：偽陽性のパラドックス
「稀な事象」に対する判定の難しさ。
* **教訓:** 検査精度が高くても、母集団の有病率が低いと、1回の陽性だけでは「ほぼ病気」とは断定できない（事後確率が低い）。
* **解決策:** 条件付き独立な「別の情報源」を足すことで、ベイズ更新により確信度を劇的に高められる。

## 5. 期待値・分散・共分散
分布を「代表値」で要約する。

* **期待値 $E[X]$:** 平均的なリターン。
* **分散 $Var[X]$:** データの散らばり（リスクの指標）。
* **共分散行列 $\Sigma$:** 複数の変数の相関関係を記述する。

[Image of a covariance matrix visualization showing correlation between different dimensions of a dataset]

## 6. 不確実性の分類
1. **アレアトリック不確実性 (Aleatoric):** * 世界の本質的なランダム性。データを増やしても消えない（例：急所、ダメージ乱数）。
2. **エピステミック不確実性 (Epistemic):**
   * モデルや知識の不足。データを集めることで減らせる（例：相手の努力値振り、選出予測）。

---

## 💡 ポケモンAI開発への応用（展望）
* **ベイズ推論:** 相手の1つ目の行動から「こだわりアイテム」の所持率を更新し、2つ目の行動（交代など）でさらに確信を強めるロジック。
* **期待値最大化:** 各ターンの行動（技・交代）による「勝利への期待値」を算出し、最も高いものを選ぶエージェントの構築。
* **大数の法則:** ダメージの振れに一喜一憂せず、長期的な勝率を最大化する戦略の評価。
