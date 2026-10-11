<img src="https://github.com/easyatmos/easyclimate-rust/blob/main/docs/logo/easyclimate_rust_logo_mini.png?raw=true" alt="easyclimate-rust">

<h2 align="center">easyclimate の Rust バックエンド</h2>


![PyPI - Version](https://img.shields.io/pypi/v/easyclimate-rust)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/easyclimate-rust)
![PyPI - Downloads](https://img.shields.io/pypi/dm/easyclimate-rust)


<div align="center">
<center><a href = "../README.md">English</a> / <a href = "README_zh_CN.md">简体中文</a> / 日本語</center>
</div>


## 🤗 easyclimate-rust とは？

**easyclimate-rust** は Rust で実装された高性能バックエンドライブラリであり、
[easyclimate](https://github.com/easyatmos/easyclimate)
における計算負荷の高い処理を担うことを目的としています。

Rust の **高い実行性能**、**メモリ安全性**、および **ゼロコスト抽象化** を活用することで、
easyclimate-rust は Python フロントエンドに対してシンプルで使いやすい API を提供しつつ、
大規模な気候データや複雑な診断計算に対しても高いスケーラビリティと効率性を実現します。

> 🚨 **プロジェクト状況：開発中** 🚨  
>
> 本プロジェクトは現在も活発に開発が進められています。
> API（関数、クラス、インターフェース）は **まだ安定しておらず**、
> 将来的に後方互換性なしで変更される可能性があります。
> 本番環境での利用には十分ご注意ください。

## 😯 インストール

`easyclimate-rust` は、Python のパッケージマネージャ
[pip](https://pip.pypa.io/en/stable/getting-started/)
を用いてインストールできます。

```bash
pip install easyclimate-rust
````

## ✨ 動作要件

* **Python** ≥ 3.10
* **NumPy** ≥ 1.24.3
  *(事前ビルドされた wheel を実行時に使用する場合のみ必要)*

## 🔧 ビルド手順

### Windows

1. Rust をインストール
   👉 [Windows で Rust 用の開発環境を設定する](https://learn.microsoft.com/ja-jp/windows/dev-environment/rust/setup)

2. `uv` をインストール：

```powershell
winget install uv
```

3. プロジェクトのルートディレクトリでビルドスクリプトを実行：

```powershell
.\scripts\build_manywindows_wheel.ps1
```

4. 生成された wheel ファイルは `dist/` ディレクトリに配置されます。

### Linux

1. Docker をインストールしてください。
2. Linux 環境上でビルドスクリプトを実行します。

```bash
./scripts/build_manylinux_wheel.sh
```

生成された wheel ファイルも `dist/` ディレクトリに配置されます。


## 🪐 オープンソースソフトウェア声明

[説明文書](https://easyclimate-backend.readthedocs.io/en/latest/src/softlist.html)を参照してください。