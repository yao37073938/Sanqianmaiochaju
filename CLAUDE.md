# 项目：线粒体网络临界相变

## 背景
这是一个计算生物学研究项目，目标是用数学模型预测线粒体网络在损伤压力下的
临界相变（鞍结分岔），并设计实验验证方案。

## 技术栈
- Python 3.10+
- numpy, scipy, matplotlib, SALib, multiprocessing
- 所有图表用 matplotlib，出版质量（300 dpi PDF）
- 代码风格：清晰的函数文档、类型注解

## 项目结构
- models/：动力学模型（ODE, SDE, Gillespie）
- analysis/：分岔分析、早期预警、敏感性分析
- figures/：输出图表
- data/：模拟数据缓存

## 关键生物学参数范围
- k_bio：1-10（线粒体/小时）
- k_fis：0.01-0.1（/小时）
- k_fus：0.00005-0.0003（/线粒体/小时）
- k_mit：0.01-0.1（/小时）
- HeLa 细胞中线粒体数量约 500-2000
