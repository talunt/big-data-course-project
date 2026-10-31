# 数据来源

使用 UCI Machine Learning Repository 的 **Bike Sharing Dataset**：

- 数据页：https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- 下载地址：https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip
- 数据作者：Hadi Fanaee-T、João Gama。
- 内容：华盛顿 Capital Bikeshare 在 2011—2012 年的日、小时租赁记录，以及天气和日历信息。
- 本项目仅使用 `day.csv`，共 731 行。预测每日租赁总量，使用温度、湿度、风速、年份、工作日、月份、星期、天气类型；另由上述字段计算六个扩展项。
- 临时用户和注册用户数量相加得到预测目标，故只用于数据质量检查。编号、日期、季节、体感温度不作为模型输入；日期用于排序和时间划分。

下载压缩包 SHA-256：
`b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401`

`day.csv` SHA-256：
`a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be`

脚本验证文件摘要后读取固定数据。数据原始说明随下载缓存保留。

相关论文：Fanaee-T, H.; Gama, J. *Event labeling combining ensemble detectors and background knowledge*. DOI: https://doi.org/10.1007/s13748-013-0040-3
