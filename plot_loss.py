import matplotlib.pyplot as plot
def getLoss(logFile):
    f = open(logFile, "r", encoding='utf-8')
    line = f.readline()  # 以行的形式进行读取文件
    lossArr = []
    lossArr_train = []

    while line:
        if 'Epoch' in line:
            lossArr.append(float(line[-7:]))
            lossArr_train.append(float(line[-44:-38]))
        line = f.readline()
    f.close()
    return lossArr,lossArr_train

def drawLine(arr, xName, yName, title, graduate,label):
    # 横坐标 采用列表表达式
    x = [x+1 for x in range(len(arr))]
    # 纵坐标
    y = arr
    # 生成折线图：函数polt
    plot.plot(x, y,label=label)
    # 设置横坐标说明
    plot.xlabel(xName)
    # 设置纵坐标说明
    plot.ylabel(yName)
    # 添加标题
    plot.title(title)
    # 设置纵坐标刻度
    plot.yticks(graduate)
    # 显示网格
    plot.grid(True)
    # 显示图表
    plot.legend()
    plot.savefig(fname="line128.png")
if __name__ == '__main__':
    loss,loss_train= getLoss(r"record_scale128.log")
    # print(loss)

    graduate = []
    deGraduate = 1.5
    # 计算y的刻度值
    for i in range(len(loss)):

        print(i)
        if i * deGraduate < max(loss) + deGraduate:
            graduate.append(i * deGraduate)
    print(graduate)
    drawLine(loss, "Epoch", "Loss", "Loss function curve of VPU+GNN", graduate,'Test Loss')
    drawLine(loss_train, "Epoch", "Loss", "Loss function curve of VPU+GNN", graduate,'Train Loss')

