# UniversityResearch


 実験準備

	・使用言語：Python

	・実験条件を変更したい場合，main.pyから変更
	
	・提案手法の指定については，1000行目の"test_ecm_abc=ABC(～)"でパラメータを指定する．各種設定できるパラメータは以下の引数のようになる．
	ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
	eva_name						：評価手法の名前である文字列を与えることで評価方法を指定
	data							：データセットの情報を与える
	target							：正解ラベルを与える
	food_num(デフォルト：10)			：個体数を指定
	feature_num_min(デフォルト：2)		：特徴選択がこの値より少ない値にならないようにする
	cluster_num_max(デフォルト：10)		：クラスタの数がこの値より超過しないようにする
	sol_lim_min(デフォルト：0.0)		：選択確率の最小値
	sol_lim_max(デフォルト：1.0)		：選択確率の最大値
	feature_sele_dthr(デフォルト：0.5)	：選択確率
	mabc_run(デフォルト：False)			：MABCアルゴリズムの使用有無
	mabc_mr(デフォルト：0.5)			：MABCアルゴリズムの修正率
	ecm_dthr_min(デフォルト：1.0e-05)	：閾値の最小値
	ecm_dthr_max(デフォルト：1.0)		：閾値の最大値
	ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー

	・繰り返し回数は1004行目のtest_ecm_abc.run("繰り返し回数")で指定する．

	・PBM及びXB，DIの3つの評価手法は，RのパッケージをPythonから呼び出して使用する．
	よって，Rの開発環境も用意し，"clValid"及び"clusterCrit"，"fclust"，"frbs"の4つのパッケージをダウンロードしておくこと．

	・「dataset」フォルダには実験で使用したデータセットがあり，UCI Machine Learning Repository[1]のサイトから入手したものである．

	[1] Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
	Irvine, CA: University of California, School of Information and Computer Science.(参照 2022-
	01-25)．

	・他のデータセットを使用したい場合，「dataset」フォルダ内に入れて，DataSelectクラスを修正する．

	・他の評価手法を追加したい場合，Validationクラスに追加．


 操作手順

	1．main.pyを開き，繰り返し回数や提案手法の指定など，設定を指定しセーブ．

	2．コマンドプロンプトにて[python main.py]と入力し，実行．

	3．Endが表示されたら実験終了．「実験結果」フォルダの「(設定したプログラム名)」フォルダに，評価値などの簡易的な結果(txt)と詳細結果(txt),
	「画像」フォルダには繰り返し回数に対する評価値の推移及び各クラスタに分かれた図(特徴量が2のとき)が生成される．
