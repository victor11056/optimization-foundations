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
