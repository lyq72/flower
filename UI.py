# 使用gui界面使操作更加人性化
import tkinter
import tkinter.filedialog
import cv2
import numpy as np
from PIL import ImageTk
import tensorflow as tf
from PIL import Image
top = tkinter.Tk()
top.title('花卉识别')
top.geometry('640x480')
top.configure(bg='#FFFAFA')
global gImage
gImage = None


def showImg(img1):
    load = Image.open(img1)
    load = load.resize((250, 250), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(load)
    img = tkinter.Label(image=render)
    img.image = render
    img.place(x=70, y=70)


def choose_fiel():
    global gImage
    selectFileName = tkinter.filedialog.askopenfilename(title='请选择文件')  # 选择文件
    showImg(selectFileName)
    gImage = selectFileName
    print(gImage)


def start():
    model = tf.keras.models.load_model("models/MobileNet_fv.h5")
    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    img_init = cv2.imread(gImage)
    img_init = cv2.resize(img_init, (224, 224))
    img4 = np.asarray(img_init)  # 将图片转化为numpy的数组
    outputs =model.predict(img4.reshape(1, 224, 224, 3))
    result_index = int(np.argmax(outputs))
    w = tkinter.Label(top, text="识别结果为:"+str(class_names[result_index]),padx=4,pady=4,font=('宋体',20,'bold'))
    w.place(x=350, y=250)


load = Image.open("E:\\pycharm\\project\\flower\\split_photos\\val\\daisy\\107592979_aaa9cdfe78_m.jpg")
load = load.resize((250, 224), Image.ANTIALIAS)
render = ImageTk.PhotoImage(load)
img = tkinter.Label(image=render)
img.image = render
img.place(x=70, y=90)


submit_button = tkinter.Button(top, text="请选择文件", command=choose_fiel,padx=4,pady=4,font=('宋体',12,'bold'))
submit_button.place(x=350, y=90)

submit_button = tkinter.Button(top, text="开始识别", command=start,padx=4,pady=4,font=('宋体',12,'bold'))
submit_button.place(x=350, y=170)


top.mainloop()
