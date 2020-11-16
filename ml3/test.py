import matplotlib.pyplot as plt

if __name__ == '__main__':
    epochSize=20
    
    plt.figure(1)
    x1=range(0,epochSize)
    x2=range(0,epochSize)
    x3=range(0,epochSize)
    y1=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    y2=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    y3=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    plt.plot(x1,y1,'o-')
    plt.title('Test accuracy in epoches')
    plt.ylabel('Test accuracy')
    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(x2, y2, '.-')
    plt.title('Train and Test Loss in epoches')
    plt.ylabel('Train loss')
    plt.subplot(2, 1, 2)
    plt.plot(x3, y3, '.-')
    plt.xlabel('epoches')
    plt.ylabel('Test loss')
    plt.show()