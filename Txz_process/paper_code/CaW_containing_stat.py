import numpy as np

data_np = np.array([248,  75,  206,  691,  156,  267,  251,  92,  516,  629,  166,  278,  152,  532,  387,  413,  173,  229,  165,  178,  154,  555,  203,  254,  275,  471,  319,  218,  185,  1150,  214,  431,  113,  236,  156,  211,  310,  125,  350, 604,  1613,  241,  215,  268,  951,  344,  176,  385,  110,  169,  298,  152,  230,  116, 338, 187, 139, 298, 198, 62,  279])

# 转化为numpy数组
# 计算均值
mean = np.mean(data_np)
print("均值：", mean)

# 计算中位数
median = np.median(data_np)
print("中位数：", median)

# 计算最小值
min_val = np.min(data_np)
print("最小值：", min_val)

# 计算最大值
max_val = np.max(data_np)
print("最大值：", max_val)

# 计算方差
var = np.var(data_np)
print("方差：", var)

# 计算标准差
std = np.std(data_np)
print("标准差：", std)
