import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
# plt.ioff()
# print(matplotlib.is_interactive())
x = [1,2,3,4,5]        # 水平資料點
h = [10,20,30,40,50]   # 高度
plt.bar(x,h)
# plt.show()

# plt.ion()
plt.show()