import tkinter as tk
from tkinter import ttk, LEFT, END
#from PIL import Image , ImageTk 

import math
import pandas as pd
import numpy as np

import train_modelCL as TrainM
import time
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

##############################################+=============================================================
root = tk.Tk()
root.configure(background="seashell2")
#root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Electricity Theft Data Analysis")

##############################################+=============================================================
lbl = tk.Label(root, text="Electricity Theft Detection Using Machine Learning ( XGBOOST Classifier )", font=('times', 35,' bold '),justify=tk.LEFT, wraplength=1700 ,bg="white",fg="indian red")
lbl.place(x=10, y=5)


frame_CP = tk.LabelFrame(root, text=" Control Panel ", width=200, height=750, bd=5, font=('times', 12, ' bold '),bg="lightblue4",fg="white")
frame_CP.grid(row=0, column=0, sticky='s')
frame_CP.place(x=5, y=60)

frame_display = tk.LabelFrame(root, text=" ---Result--- ", width=1100, height=750, bd=5, font=('times', 12, ' bold '),bg="white",fg="red")
frame_display.grid(row=0, column=0, sticky='s')
frame_display.place(x=210, y=60)

frame_noti = tk.LabelFrame(root, text=" Notification ", width=250, height=750, bd=5, font=('times', 12, ' bold '),bg="lightblue4",fg="white")
frame_noti.grid(row=0, column=0, sticky='nw')
frame_noti.place(x=1330, y=60)

###########################################################################################################
canvas=tk.Canvas(frame_display,bg='#FFFFFF',width=1000,height=600,scrollregion=(0,0,2000,8000))
hbar=tk.Scrollbar(frame_display,orient=tk.HORIZONTAL)
hbar.pack(side=tk.BOTTOM,fill=tk.X)
hbar.config(command=canvas.xview)
vbar=tk.Scrollbar(frame_display,orient=tk.VERTICAL)
vbar.pack(side=tk.RIGHT,fill=tk.Y)
vbar.config(command=canvas.yview)
canvas.config(width=1100,height=680)
canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
#canvas.pack(side=LEFT,expand=True,fill=tk.BOTH)
canvas.pack(fill=tk.BOTH, expand=tk.YES)

basepath=r"D:\\Codedata\\"


###########################################################################################################
####################################################################################################################
def update_label(str_T):
    result_label = tk.Label(frame_noti, text=str_T, font=("italic", 20),justify=tk.LEFT, wraplength=200 ,bg='lightblue4',fg='white' )
    result_label.place(x=10, y=0)

def display_data():

    df=pd.read_csv(basepath + 'data.csv')
    df=df.head(500)
    
    
    icol=50
    colsp=500
    canvas.delete("all")
    canvas.create_text(5,15,fill="red",anchor="w",font="Times 16 bold",text="Consumer")
    canvas.create_line([(390, 0), (390, 10000)], fill='red', tags='grid_line_w')

    canvas.create_text(400,5,fill="red",anchor="nw",font="Times 16 bold",text="Status")
    canvas.create_line([(490, 0), (490, 10000)], fill='red', tags='grid_line_w')

    dateL=df.columns[2:20]
    print(dateL)
    
    for i in range(len(dateL)):
        
        canvas.create_text(colsp,5,fill="red",anchor="nw",font="Times 16 bold",text=dateL[i])
        colsp=colsp+100

    
    for i in range(len(df)):
        
        canvas.create_text(5,icol,fill="blue",anchor="w",font="Times 15",text=df["CONS_NO"][i])
        canvas.create_text(410,icol,fill="blue",anchor="nw",font="Times 15",text=df["FLAG"][i])
        
        colsp=500

        for j in range(len(dateL)):
            canvas.create_text(colsp,icol,fill="blue",anchor="nw",font="Times 15",text=df[dateL[j]][i])
            colsp=colsp+100
#        canvas.create_text(100,icol,fill="blue",anchor="nw",font="Times 15",text=df["2014/01/01"][i])
        
        
        canvas.create_line([(0, icol-12), (10000, icol-12)], fill='black', tags='grid_line_h')
    
        icol=icol+30
        
##################################################################################################################
def Process_data():
    
    basepath=r"D:\\Codedata\\"


    df=pd.read_csv(basepath + 'data.csv')
    ##############################################################################
    l=df.columns
    la=['CONS_NO','FLAG']
    lb=[]
    for i in l:
        if i not in la:
            lb.append(i)
            
            
    import datetime
    dates = [datetime.datetime.strptime(ts, "%Y/%m/%d") for ts in lb]
    #dates.sort()
    fdates = [datetime.datetime.strftime(ts, "%Y/%m/%d") for ts in dates]
    
    fdates.insert(0,"FLAG")
    fdates.insert(0,"CONS_NO")
    df.columns=fdates
    
    
    
    import datetime
    dates = [datetime.datetime.strptime(ts, "%Y/%m/%d") for ts in lb]
    dates.sort()
    sorteddates = [datetime.datetime.strftime(ts, "%Y/%m/%d") for ts in dates]
    
    
    
    cols=df.columns.tolist()[0:2]+sorteddates
    df=df[cols]
    print(df[cols])
    print(df)
    ######Process NAN value======================================================
    
    ####For first two col
    df1=df
    
    l=df["2014/01/01"]
    l1=df["2014/01/02"]
    l=np.asarray(l).tolist()
    l1=np.asarray(l1).tolist()
    
    l2=[]
    for i in range(len(l)):
        if math.isnan(l[i]):
            if math.isnan(l1[i]):
                l2.append(0)
            else:
                l2.append(l1[i]/2)
        else:
            l2.append(l[i])
            
    df1["2014/01/01"]=l2
    
    #############################################################################
    ####for last two days
    
    print(df.columns[-1],df.columns[-2])
    
    l=df["2016/10/31"]
    l1=df["2016/10/30"]
    l=np.asarray(l).tolist()
    l1=np.asarray(l1).tolist()
    
    l2=[]
    for i in range(len(l)):
        if math.isnan(l[i]):
            if math.isnan(l1[i]):
                l2.append(0)
            else:
                l2.append(l1[i]/2)
        else:
            l2.append(l[i])
    df1["2016/10/31"]=l2
    
    #############################################################################
    #for rest of the columns
    
    l=df.columns
    la=['CONS_NO','FLAG']
    lb=[]
    for i in l:
        if i not in la:
            lb.append(i)
    
    l=df.columns
    la=['CONS_NO','FLAG']
    lbx=[]
    for i in l:
        if i not in la:
            lbx.append(i)
          
    
    for i in range(1,len(lb)-1):
        l=np.asarray(df[lb[i]]).tolist()
        l1=np.asarray(df[lb[i-1]]).tolist()
        l2=np.asarray(df[lb[i+1]]).tolist()
        l3=[]
        for j in range(len(l)):
            if math.isnan(l[j]):
                if math.isnan(l1[j])==False and math.isnan(l2[j])==False:
                    l3.append((l1[j]+l2[j])/2)
                else:
                    l3.append(0)
            else:
                l3.append(l[j])
        df1[lb[i]]=l3
#======================================================================================================================================
    
    icol=50
    colsp=500
    df=df1.head(100)
    canvas.delete("all")
    canvas.create_text(5,15,fill="red",anchor="w",font="Times 16 bold",text="Consumer")
    canvas.create_line([(390, 0), (390, 10000)], fill='red', tags='grid_line_w')

    canvas.create_text(400,5,fill="red",anchor="nw",font="Times 16 bold",text="Status")
#    canvas.create_line([(490, 0), (490, 10000)], fill='red', tags='grid_line_w')

    dateL=df.columns[2:20]
    print(dateL)
    
    for i in range(len(dateL)):
        
        canvas.create_text(colsp+5,5,fill="red",anchor="nw",font="Times 16 bold",text=dateL[i])
        colsp=colsp+100

    
    for i in range(len(df)):
        
        canvas.create_text(5,icol,fill="blue",anchor="w",font="Times 15",text=df["CONS_NO"][i])
        canvas.create_text(410,icol,fill="blue",anchor="nw",font="Times 15",text=df["FLAG"][i])
        
        colsp=500

        for j in range(len(dateL)):
            canvas.create_text(colsp+5,icol,fill="blue",anchor="nw",font="Times 15",text=df[dateL[j]][i])
            canvas.create_line([(colsp+2, 0), (colsp+2, 10000)], fill='red', tags='grid_line_w')

            colsp=colsp+100
#        canvas.create_text(100,icol,fill="blue",anchor="nw",font="Times 15",text=df["2014/01/01"][i])
        
        
        canvas.create_line([(0, icol-12), (10000, icol-12)], fill='black', tags='grid_line_h')
    
        icol=icol+30
        
        
        
###&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    counts = df1['FLAG'].value_counts()
    print(counts[0], counts[1])
    
    lbl3 = tk.Label(frame_noti, text="Total No. of observation----"  + str(len(df1)), font=('times',20,' bold '),justify=tk.LEFT, wraplength=200  ,bg="lightblue4",fg="cyan")
    lbl3.place(x=10, y=100)

    lbl1 = tk.Label(frame_noti, text="No. of Theft observation--- "+ str(counts[1]), font=('times', 20,' bold '),justify=tk.LEFT, wraplength=200,bg="lightblue4",fg="white")
    lbl1.place(x=10, y=200)
    
    lbl2 = tk.Label(frame_noti, text="No. of Normal observation--- "+ str(counts[0]), font=('times', 20,' bold ') ,justify=tk.LEFT, wraplength=200,bg="lightblue4",fg="white")
    lbl2.place(x=10, y=300)
    

#======================================================================================================================================   
def result_data():

    Rdf=pd.read_csv(basepath + 'Result.csv')
#    Rdf=Rdf.head(30)
    
    icol=50
    canvas.delete("all")
    canvas.create_text(5,15,fill="red",anchor="w",font="Times 16 bold",text="Consumer")
    canvas.create_line([(390, 0), (390, 10000)], fill='red', tags='grid_line_w')

    canvas.create_text(400,5,fill="red",anchor="nw",font="Times 16 bold",text="Actual Observation")
    canvas.create_line([(690, 0), (690, 10000)], fill='red', tags='grid_line_w')

    canvas.create_text(700,5,fill="red",anchor="nw",font="Times 16 bold",text="Predicted Observation")
#    canvas.create_line([(490, 0), (490, 10000)], fill='red', tags='grid_line_w')

    Rdf=Rdf[['CONS_NO','FLAG','theft']]
    
    for i in range(len(Rdf)):

        canvas.create_text(5,icol,fill="blue",anchor="w",font="Times 15",text=Rdf["CONS_NO"][i])
        canvas.create_text(410,icol,fill="blue",anchor="nw",font="Times 15",text=Rdf["FLAG"][i])
        canvas.create_text(700,icol,fill="blue",anchor="nw",font="Times 15",text=Rdf["theft"][i])
       
        canvas.create_line([(0, icol-12), (10000, icol-12)], fill='black', tags='grid_line_h')
    
        icol=icol+30

def train_model():


    update_label("Model Training Start...............")
    
    start = time.time()
    StrVal= TrainM.main()
    result_data()
    end = time.time()
    
    ET="Execution Time: {0:.4} seconds \n".format(end-start)
    
#    print(Xdf.head())
    
    msg="Model Training Completed.."+'\n'+ StrVal + '\n'+ ET
    
    update_label(msg)





##########################################################################################################################################################
      
        
def window():
    root.destroy()


button1 = tk.Button(frame_CP, text=" Load Data ", command=display_data,width=19, height=1, font=('times', 12, ' bold '),bg="white",fg="black")
button1.place(x=5, y=50)

button2 = tk.Button(frame_CP, text=" Data Processing ", command=Process_data,width=19, height=1, font=('times', 12, ' bold '),bg="white",fg="black")
button2.place(x=5, y=150)
#Analysis Electricity Theft data
button3 = tk.Button(frame_CP, text=" Model Training ", command=train_model,width=19, height=1, font=('times', 12, ' bold '),bg="white",fg="black")
button3.place(x=5, y=250)


exit = tk.Button(frame_CP, text="Exit", command=window, width=19, height=1, font=('times', 12, ' bold '),bg="red",fg="white")
exit.place(x=5, y=550)



root.mainloop()