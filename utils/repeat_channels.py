
def repeat_channels(x):
    # 如果輸入的通道數為1，則重複三次以形成三通道圖像
    return x.repeat(3, 1, 1) if x.size(0) == 1 else x