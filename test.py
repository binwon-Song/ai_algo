import numpy as np
import matplotlib.pyplot as plt


# 각 막대 위에 실제 값을 표시하는 함수
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
        
        
# 데이터
categories = ['cache miss']
_64k_values = [983561049.8]
_4k_values = [1035619870]

# 막대의 위치 설정
x = np.arange(len(categories))

# 막대의 너비 설정
width = 0.4

# 그래프 크기 설정
plt.figure(figsize=(8, 10))
plt.yticks(fontsize=1)
# 막대그래프 생성
fig, ax = plt.subplots()

bars1 = ax.bar(x, _4k_values, width, label='4K', hatch='//', color='lightgray', edgecolor='black')  # 4K 막대
bars2 = ax.bar(x + width, _64k_values, width, label='64K', hatch='xx', color='darkgray', edgecolor='black')  # 64K 막대

# 그래프 제목
ax.set_title('64K and 4K by cache miss')

# y축 레이블
ax.set_ylabel('Number of cache miss')

# x축 눈금 설정
ax.set_xticks(x + width / 2)
ax.set_xticklabels(categories)

# y축 눈금 설정
ax.set_yticks(np.arange(0, 2000000000, 200000000))
ax.set_yticks(np.arange(0, 2000000000, 200000000))



# 막대 위에 실제 값을 표시
autolabel(bars1)
autolabel(bars2)

# 범례 추가
ax.legend()
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
# 그래프 표시
plt.gca().ticklabel_format(style='plain', axis='y')
plt.show()
