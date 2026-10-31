# 相关资源

## 数据

本项目通过 UCI 官方地址下载 Bike Sharing Dataset，保留原始数据说明。作者、论文及文件摘要见 [数据来源](data/SOURCE.md)。使用、引用和再分发数据时请遵循数据提供方公布的条款。

## 运行依赖

Python 依赖版本记录于 `requirements.txt`，分别为 Apache Spark / PySpark、Matplotlib、NumPy 和 setuptools。各依赖遵循其原有许可证。

Windows 运行时从以下第三方仓库下载 Hadoop 3.3.5 的 `winutils.exe` 和 `hadoop.dll`：
https://github.com/cdarlint/winutils/tree/master/hadoop-3.3.5/bin

文件不随本仓库分发。脚本内固定 SHA-256，下载后及每次运行前均核对摘要，来源记录保存在本地 `runtime/hadoop/manifest.json`。

本仓库尚未指定源代码开源许可证；公开托管前可由作者决定并补充 `LICENSE`。
