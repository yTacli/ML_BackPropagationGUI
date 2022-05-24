# Yücel TACLI

import math
from ssl import DER_cert_to_PEM_cert
import time
from pathlib import Path
from tkinter.tix import Select
import PySimpleGUI as sg
import numpy as np
import pandas as pd
import Models as md

tableValues = []   # tablo values
tableHeadings = [] # table values
epoc = "0"

VIEWWEIGHT:int = 500
VIEWHEIGHT:int = 300
config = {
    "theme" : "DarkGrey9",   
}
def create_main_window():  
    sg.theme(config["theme"])   
    file_select_layout = [          
        [sg.Frame("Dosya", [
            [
                sg.Text("Dosya Seçimi:"), 
                sg.Input(expand_x=True, key="-SELECTFILE-", enable_events=True,size=(10,1)),
                sg.FileBrowse(button_text="Dosya Seç", key="-SELECTFILE-",file_types=(('TXT Files', '*.txt'),),)
            ],            
        ], expand_x=True)],
        [sg.Frame("Seçilen Dosya Bilgileri", [
            [sg.Text("Dosya Adı : "), sg.Text("",key="-SELECTEDFILE-"), sg.Push()],
            [sg.Text("Input Sayısı : "), sg.Text("",key="-NUMBEROFINPUT-"), sg.Push()],
            [sg.Text("Output Sayısı/Adları : "),sg.Text("",key="-NUMBEROFOUT-"), sg.Push()],
            [sg.Text("Satır Sayısı : "),sg.Text("",key="-NUMBEROFLINES-"), sg.Push()]
        ],expand_x=True)], 
        [sg.Frame("Model Bilgileri", [
            [sg.Text("Gizli Katman Sayısı :   "),
                sg.Spin([x for x in range(21)],initial_value=1,  key="-HIDDENN-", size=(6,1)),
                sg.Button("Detay Bilgileri",key="-HIDDENDETAIL-"),
            ],                      
        ],expand_x=True)],
        [sg.Frame("Model Detay Bilgileri", [], expand_x=True, expand_y=True, key="-HIDDENDETAILVALUE-", visible=False)],        
        [sg.Frame("Veriler",
            [],expand_x=True, expand_y=True, key="-TABLE-")
        ],
    ]
    values_input_layout = [                                        
        [sg.Frame("Model Bilgileri", [
            [sg.Text("Giriş Katmanı Nöron Sayısı(max:20) :  "),
                sg.Spin([x for x in range(21)],initial_value=2,  key="-SECINPUTNUMBER-", size=(6,1))
            ],  
            [sg.Text("Gizli Katman Sayısı :   "),
                sg.Spin([x for x in range(21)],initial_value=1,  key="-SECHIDDENNUMBER-", size=(6,1))
            ],    
            [sg.Text("Çıkış Katmanı Nöron Sayısı(max:20) :   "),
                sg.Spin([x for x in range(21)],initial_value=2,  key="-SECOUTPUTNUMBER-", size=(6,1))
            ],                  
        ],expand_x=True)],
        [sg.Button("Detay Bilgileri",key="-INPUTDETAIL-")],
        [sg.Frame("Model Detay Bilgileri", [], expand_x=True, expand_y=True, key="-SECMODELVALUE-", visible=False)],
    ]
    graph_layout = [
        [sg.Graph(
            canvas_size=(VIEWWEIGHT,VIEWHEIGHT),
            graph_bottom_left=(0,0),
            graph_top_right=(VIEWWEIGHT,VIEWHEIGHT),
            background_color="white",
            key="graph",
            enable_events = True,
            drag_submits = True         
        )],
        [ 
            sg.Button(key="-NEXT-",image_source="images/btn_next_ani.png",image_subsample=2,disabled=True),
            sg.Push(),
            sg.Button(key="-BACK-",image_source="images/btn_prev_ani.png",image_subsample=2,disabled=True),
            sg.Push(),
            sg.Button(key="-PLAY-",image_source="images/btn_play_ani.png",image_subsample=2,disabled=True),
        ],
        [sg.Frame("Hatalar",[
            [sg.Text("SSE : "),sg.Text("",key="-SSE-")],
            [sg.Text("MSE : "),sg.Text("",key="-MSE-")],
            [sg.Text("RMSE : "),sg.Text("",key="-RMSE-")]            
        ],expand_x=True,expand_y=True)]       
    ]    
    out_data_column = [   
        [sg.Listbox([],size=(40,26),key="-LISTBOX-")]
        # [sg.Output(size=(30,30))], #print komutu kullanılabiliyor          
    ]
    main_layout = [
        [sg.Push(), sg.Text("Machine Learning Back Propagation", font=(sg.DEFAULT_FONT,20)), sg.Push()],
        [
            sg.pin(sg.TabGroup([[
                sg.Tab("Veri Giriş", values_input_layout), 
                sg.Tab("Dosya Seçim", file_select_layout)
            ]],enable_events=True,key="-TAB-",size=(400,430)), vertical_alignment = "top"),
            sg.pin(sg.Column(graph_layout),vertical_alignment= "top"),            
            sg.Column(out_data_column,vertical_alignment = "top")
        ],
        [
            sg.Text("Öğrenme Oranı : "),
            sg.Spin([x for x in np.arange(0.0, 1.01, 0.01)], initial_value=0.05,  key="-LEARNRATE-", size=(6,1)),            
            sg.Text("Aktivasyon Fonksiyonu : "),
            sg.Combo(['Sigmoig','Threshold'],'Sigmoid',auto_size_text=True,key='-ACTIVATION-'),
            sg.Text("Threshold : "),
            sg.Spin([x for x in np.arange(0.0, 1.01, 20.0)],initial_value=0.0,  key="-THRESHOLD-", size=(6,1))            
        ],
        [
            sg.Button("Modeli Oluştur", key="-CREATEMODELVIEW-",disabled=True),
            sg.Button("Başlangıç Değerleri", key="-INITIALIZE-",disabled=True),
            sg.Push(),
            sg.Button("RESET", key="-RESET-")
        ],
        [sg.StatusBar("Yücel TACLI")]
    ]
    window = sg.Window("Back Propataion GUI", layout=main_layout, finalize=True)
    return window

def create_model_view(input,hidden,output):
    # input:int
    # hidden = [3,2,5]
    # output:int
    g = main_window["graph"]
    #input çiz
    inputDist = VIEWHEIGHT/(input+1)
    inputY=[]
    inputY.append(inputDist)
    for i in range(input):
        g.draw_line((20,inputDist), (35,inputDist), width=2)        
        g.draw_point((50,inputDist), 30, color="green"),
        g.draw_text(i+1, (50,inputDist), color = "white")
        inputDist += VIEWHEIGHT/(input+1)
        inputY.append(inputDist)

    # hidden çiz    
    hiddenXDist = 80
    hiddenX = []
    hiddenY = []
    for i in range(len(hidden)):
        hInY = []
        hiddenXDist += (VIEWWEIGHT-160)/(len(hidden)+1)
        hiddenX.append(hiddenXDist)
        hiddenYDist = 0       
        for j in range(hidden[i]):
            hiddenYDist += VIEWHEIGHT/(hidden[i]+1)
            hInY.append(hiddenYDist)
            g.draw_point((hiddenXDist, hiddenYDist), 30, color="purple")   
            g.draw_text(j+1, (hiddenXDist,hiddenYDist), color = "white")
        hiddenY.append(hInY)     

    #output çiz
    outDist = VIEWHEIGHT/(output+1)
    outY = []
    outY.append(outDist)
    for i in range(output):
        g.draw_line((VIEWWEIGHT-35,outDist),(VIEWWEIGHT-20,outDist), width=2)
        g.draw_point((VIEWWEIGHT-50, outDist), 30, color="orange")  
        g.draw_text(i+1, (VIEWWEIGHT-50, outDist), color = "black")
        
        outDist += VIEWHEIGHT/(output+1)
        outY.append(outDist)

    #bias çiz
    biasDist = 50 + (hiddenX[0]-50)/2
    biasX = []  
    biasX.append(biasDist) 
    g.draw_point((biasDist, VIEWHEIGHT-20), 30, color="gray")        
    g.draw_text("b1", (biasDist,VIEWHEIGHT-20), color="black") 
    for i in range(1,len(hidden)):         
        biasDist = hiddenX[i-1] + ((hiddenX[i] - hiddenX[i-1])/2)
        g.draw_point((biasDist, VIEWHEIGHT-20), 30, color="gray")        
        g.draw_text("b"+str(i+1), (biasDist,VIEWHEIGHT-20), color="black")         
        biasX.append(biasDist) 
    biasDist = (hiddenX[len(hidden)-1]) + (((VIEWWEIGHT-50)-(hiddenX[len(hidden)-1]))/2)
    biasX.append(biasDist)
    g.draw_point((biasDist, VIEWHEIGHT-20), 30, color="gray")
    g.draw_text("b"+str(len(hidden)+1), (biasDist,VIEWHEIGHT-20), color="black")     
    # ağırlık çiz
    for i in range(input):
        for j in range(hidden[0]):
            g.draw_line((65,inputY[i]),(hiddenX[0]-15,hiddenY[0][j]),width=2)         
    for i in range(1,len(hidden)):
        for j in range(hidden[i-1]):
            for k in range(hidden[i]):
                g.draw_line((hiddenX[i-1]+15,hiddenY[i-1][j]),(hiddenX[i]-15, hiddenY[i][k]),width=2)
    for i in range(hidden[len(hidden)-1]):
        for j in range(output):
            g.draw_line((hiddenX[len(hidden)-1]+15,hiddenY[len(hidden)-1][i]),(VIEWWEIGHT-65, outY[j]),width=2)  
    #bias ağırlık çiz         
    for i in range(len(hidden)):
        for j in range(hidden[i]):
            g.draw_line((biasX[i]+10,VIEWHEIGHT-30),(hiddenX[i]-15,hiddenY[i][j]),width=1,color="gray")
    for i in range(output):
        g.draw_line((biasX[len(hidden)]+10,VIEWHEIGHT-30),(VIEWWEIGHT-65,outY[i]), width=1, color="gray")   
    # #g.DrawImage(filename = "images/forward.png",location = (250, 50)) 

# iconlar görünmüyor
def forward_icon():
    g = main_window["graph"]
    g.draw_image(filename = "images/forward.png",location = (250, 50))
    main_window.Refresh()
def backward_icon():
    g = main_window["graph"]
    g.draw_image(filename = "images/backward.png",location = (250, 50))
    main_window.Refresh()

def select_detail_input(inputNumber,hiddenLayerNumber,outputNumber):  
    inp_hidden =[        
        [
            sg.Text(f"h{x} : "),
            sg.Input(default_text=0, key=f"-sh{x}-", size=(10,1),justification="center")
        ] for x in range(1,hiddenLayerNumber+1)
    ]
    inp_col = [            
            [
                sg.Text(f"i{x} : "),
                sg.Input(default_text=0.0, key=f"-i{x}-", size=(10,1),justification="center")
            ] for x in range(1,inputNumber+1)
    ]    
    out_col = [
            [
                sg.Text(f"o{x} : "),                
                sg.Input(default_text=0.0, key=f"-o{x}-", size=(10,1),justification="center")
            ] for x in range(1,outputNumber+1)
    ]
    layout = [
        [            
            sg.Column(inp_hidden),
            sg.Column(inp_col),            
            sg.Column(out_col),
        ],        
    ]
    return layout
def hidden_detail_input(hiddenLayerNumber):
    # 3'lü gruplarda sg.column yapılabilir.
    inp_hidden = [        
        [
            sg.Text(f"Hidden-{x}-Noron Sayısı : "),
            sg.Input(default_text=0, key=f"-fh{x}-", size=(10,1),justification="center")
        ] for x in range(1,hiddenLayerNumber+1)
    ]
    layout = [[sg.Column(inp_hidden)]]
    return layout
def file_row_select(dataFrame,rowsNumber):
    selectRow = list(dataFrame.loc[rowsNumber])
    selectRowInput = [] 
    for i in range(1,len(selectRow)):               
        selectRowInput.append(int(selectRow[i])) # verilerde string varsa

    # 2 output için
    if selectRow[len(selectRow)] == 0:
        out2 = 1
    else:
        out2 = 0
    selectRowOutput = []
    selectRowOutput.append(selectRow[len(selectRow)])
    selectRowOutput.append(out2)
    return selectRowInput,selectRowOutput

main_window = create_main_window()

def create_detay_window(dataFrame):    
    sg.theme(config["theme"]) 
    columnsName = list(dataFrame.columns)  
      
    print(columnsName)
    # tablodaki ilk veri satırı 
    firstRow = list(dataFrame.loc[0])
    tableValues.append(firstRow)  
    # tablodaki veri türleri
    typ = []
    for col in firstRow:
        if type(col) is np.float_:
            typ.append("float")
        elif type(col) is np.int64:
            typ.append("int")
        elif type(col) is str:
            typ.append("str")
        tableValues.append(typ)

    dataTable_layout = [          
         [sg.Frame("Veriler",
            [
                sg.Table(values=tableValues, headings=columnsName, expand_x=True, expand_y=True, key="-DATATABLE-")
            ],expand_x=True, expand_y=True, key="-TABLE-"),           

        ],
    ]
    select_Layout =[
        [sg.Frame("Veriler",
            [
                [
                    sg.Text(f"{column} : "),
                    sg.Combo(["ID","INPUT","OUTPUT"],"ID",auto_size_text=True,key=f"-{column}-SELECT-")
                ] for column in columnsName
            ]
        )]
    ]
    main_layout = [
        [dataTable_layout], 
        [select_Layout],        
        [sg.Button("Devam",key="-SELECTCONTINUE-")]
    ]    
    window = sg.Window("VERİ SEÇİM", layout=main_layout, finalize=True)
    return window

while 1:
    window, event, values = sg.read_all_windows()  
    selectedTab = main_window["-TAB-"].get()
    if window is None:
        break    
    if event in (sg.WIN_CLOSED, "Exit"):
        window.close()
        main_window.close()   
        if window != main_window: 
            main_window.close()
            break  
    elif event == "-INPUTDETAIL-":         
        if selectedTab == "Veri Giriş":
            secInputNumber = int(main_window['-SECINPUTNUMBER-'].get())
            secHiddenLayerNumber = int(main_window['-SECHIDDENNUMBER-'].get())
            secOutputNumber = int(main_window['-SECOUTPUTNUMBER-'].get())            
            if secInputNumber <= 0 or secHiddenLayerNumber <= 0 or secOutputNumber <=0:
                sg.popup("Değerler SIFIR veya sıfırdan küçük olamaz olamaz")
            else:
                main_window.extend_layout(window['-SECMODELVALUE-'], select_detail_input(secInputNumber,secHiddenLayerNumber,secOutputNumber))                  
            main_window["-INPUTDETAIL-"].update(disabled = True)
            main_window["-SECMODELVALUE-"].update(visible = True)
            main_window["-INPUTDETAIL-"].update(disabled = True)
            main_window["-CREATEMODELVIEW-"].update(disabled = False) 
    elif event == "-HIDDENDETAIL-":
        if selectedTab == "Dosya Seçim":
            fileHiddenLayerNumber = int(main_window["-HIDDENN-"].get())
            main_window.extend_layout(window['-HIDDENDETAILVALUE-'], hidden_detail_input(fileHiddenLayerNumber))
            main_window["-HIDDENDETAIL-"].update(disabled = True)
            main_window["-HIDDENDETAILVALUE-"].update(visible = True) 
            main_window["-CREATEMODELVIEW-"].update(disabled = False)
    elif event == "-SELECTFILE-":
        fileName = values["-SELECTFILE-"]        
        try:            
            df = pd.read_table(fileName, delimiter="\t")
            rowsNumber = df.shape[0] - 1    # satır sayısı (başlıkları sayma) 
            # Buradan sonra yeni form ile input, output column seçimi yapılabilir
            # select_window = create_detay_window(df) #Dynamic oluşturma hatası veriyor!!!

            fileInputNumber = df.shape[1] - 2   # ilk ve  son sütunu(output) sayma
                        
            # output gruplama
            outGrup = df.groupby('Class')   # Class breast_cancer_data için geçerli sütun adı globalleşmesi gerekiyor           
            fileOutputNumber = len(outGrup.groups)  # class sayısı            
            outstr = outGrup.groups.keys()  # class isimleri           
            outputString = str(fileOutputNumber) + " / " + str(list(outstr))          
           
            tableHeadings = list(df.columns)            
            firstRow = list(df.loc[0])
            tableValues.append(firstRow)
            column = list(df.loc[0])

            # tablodaki veri türleri         
            typ = []
            for col in column:
                if type(col) is np.float_:
                    typ.append("float")
                if type(col) is np.int64:
                    typ.append("int")
                if type(col) is str:
                    typ.append("str")
            tableValues.append(typ)

            main_window["-SELECTEDFILE-"].update(Path(fileName).stem)
            main_window["-NUMBEROFINPUT-"].update(fileInputNumber)
            main_window["-NUMBEROFOUT-"].update(outputString) 
            main_window["-NUMBEROFLINES-"].update(rowsNumber)
            main_window["-CREATEMODELVIEW-"].update(disabled = True) 
            # main_window.extend_layout(main_window['-TABLE-'], data_tablo_load(values,headings))  
            main_window.extend_layout(main_window['-TABLE-'], [[sg.Table(tableValues,tableHeadings, expand_x=True, expand_y=True, key="-DATATABLE-")]])
        except:
            pass                    
    elif event == "-CREATEMODELVIEW-": 
        if selectedTab == "Veri Giriş":
            secHiddenList = []   
            for i in range(secHiddenLayerNumber):
                secHiddenList.append(int(main_window["-sh"+str(i+1)+"-"].get()))                
            create_model_view(secInputNumber,secHiddenList,secOutputNumber)                        
            inputValues = []
            outputTarget = []
            for i in range(secInputNumber):           
                inputValues.append(float(main_window["-i"+str(i+1)+"-"].get()))           
            for i in range(secOutputNumber):
                outputTarget.append(float(main_window["-o"+str(i+1) + "-"].get()))
            main_window["-CREATEMODELVIEW-"].update(disabled = True)            
           
        else: 
            fileHiddenList = []
            for i in range(fileHiddenLayerNumber):
                fileHiddenList.append(int(main_window["-fh"+str(i+1)+"-"].get())) 
            create_model_view(fileInputNumber,fileHiddenList,fileOutputNumber)    
            # rowsNumber, tempdatas var
            tempdatas = df.copy(deep=True)  # datanın kopyasını oluştur. İşlemler bunun üzerinden yapılacak

            # temp datalarda nominal dataları sayısal yapma             
            # breast_cancer datası için output sütununa direk(1 ve 0) değer atandı
            sonsutun = tempdatas.columns[fileInputNumber+1]
            for i in range(fileOutputNumber):
                for j in range(rowsNumber):
                    if tempdatas.iloc[j][fileInputNumber+1] == list(outstr)[i]:
                        tempdatas.at[j,sonsutun] = i
            fileInput, outputTarget =  file_row_select(tempdatas,0)     # ilk sutun
           
        main_window["-CREATEMODELVIEW-"].update(disabled = True) 
        main_window["-INITIALIZE-"].update(disabled = False)       
    elif event == "-INITIALIZE-":     
        activationFunction = values['-ACTIVATION-']
        threshold = float(main_window["-THRESHOLD-"].get())
        learningRate = float(main_window['-LEARNRATE-'].get())     

        if selectedTab == "Veri Giriş":
            model,weights,bias = md.create_model_base(secInputNumber,inputValues,secHiddenLayerNumber,secHiddenList,secOutputNumber,activationFunction,threshold)            
        else:
            model,weights,bias = md.create_model_base(fileInputNumber,fileInput,fileHiddenLayerNumber,fileHiddenList,fileOutputNumber,activationFunction,threshold)        

        printVal = ["WEIGHT"]        
        for layer in range(len(weights)):
            for noron in range(len(weights[layer])):
                for nextNoron in range(len(weights[layer][noron])): # next noron
                    printVal.append("w"+str(layer+1)+"_"+str(noron+1)+"-"+str(nextNoron+1)+"= "+str(weights[layer][noron][nextNoron]))
        printVal.append("BIAS")        
        for b in range(len(bias)):
            for nextNoron in range(len(bias[b])):
                printVal.append("b"+str(b+1)+"-"+str(nextNoron+1)+"= "+str(bias[b][nextNoron]))
                
        main_window["-LISTBOX-"].update(values=printVal)            
        main_window["-INITIALIZE-"].update(disabled = True) 
        main_window["-NEXT-"].update(disabled = False) 
        main_window["-PLAY-"].update(disabled = False)
    elif event == "-NEXT-":
        fmodel= md.forward(model,weights,bias)         
        printVal = []        
        for i in range(len(fmodel[0].norons)):
            printVal.append("i"+str(i+1)+"= "+str(fmodel[0].norons[i].value))
        for layer in range(1,len(fmodel)):
            for noron in range(len(fmodel[layer].norons)):
                printVal.append("h"+str(layer)+"-"+str(noron+1)+"= "+str(fmodel[layer].norons[noron].value))
        for noron in range(len(fmodel[len(fmodel)-1].norons)):
                printVal.append("o"+str(noron+1)+"= "+str(fmodel[len(fmodel)-1].norons[noron].value))

        main_window["-LISTBOX-"].update(values=printVal)        
        sse = md.sum_square_error(fmodel,outputTarget)        
        main_window["-SSE-"].update(value=sse)
        mse = md.mean_square_error(fmodel,outputTarget)
        main_window["-MSE-"].update(value=mse)
        rmse = md.root_mean_square_error(fmodel,outputTarget)
        main_window["-RMSE-"].update(value=rmse)
        
        main_window["-NEXT-"].update(disabled = True)              
        main_window["-BACK-"].update(disabled = False)        
    elif event == "-BACK-":    
        bmodel,upW,upB =md.backward(fmodel,weights,bias,outputTarget,learningRate)
        printVal = []
        for layer in range(len(weights)):
            for noron in range(len(weights[layer])):
                for nextNoron in range(len(weights[layer][noron])): # next noron
                    printVal.append("w"+str(layer+1)+"_"+str(noron+1)+"-"+str(nextNoron+1)+"= "+str(weights[layer][noron][nextNoron]))
                    printVal.append("up_w"+str(layer+1)+"_"+str(noron+1)+"-"+str(nextNoron+1)+"= "+str(upW[layer][noron][nextNoron]))
        printVal.append("BIAS")        
        for b in range(len(bias)):
            for nextNoron in range(len(bias[b])):
                printVal.append("b"+str(b+1)+"-"+str(nextNoron+1)+"= "+str(bias[b][nextNoron]))
                printVal.append("up_b"+str(b+1)+"-"+str(nextNoron+1)+"= "+str(upB[b][nextNoron]))

        model = bmodel
        weights = upW
        bias = upB

        main_window["-LISTBOX-"].update(values=printVal)
        main_window["-NEXT-"].update(disabled = False)              
        main_window["-BACK-"].update(disabled = True) 
        main_window["-PLAY-"].update(disabled = False) 
    elif event == "-PLAY-":         
        while epoc == '0' or epoc == "":
            epoc = sg.popup_get_text('epoc', "EPOC SAYISI",'0')        
            if epoc == '0' or epoc == "":
                sg.popup("Epoc 0 veya boş olamaz!!!")       
        # İlklendirildiği için aynı sonucu verecektir. 
        # Döngü Backward ile başlayacağı için ilk değer olmazsa hata verecektir.
        fmodel= md.forward(model,weights,bias)
        
        if selectedTab == "Dosya Seçim":
            for i in range(rowsNumber):
                fileInput, outputTarget =  file_row_select(tempdatas,rowsNumber)     # ilk sutun               
                for i in range(len(fmodel[0].norons)):
                    fmodel[0].norons[i].value = fileInput[i]    # input
                for j in range(len(fmodel[len(fmodel)])):
                    fmodel[len(fmodel)].norons[j].value = outputTarget[j]  # output

                for i in range (int(epoc)):        
                    bmodel,upW,upB =md.backward(fmodel,weights,bias,outputTarget,learningRate)  
                    fmodel= md.forward(bmodel,upW,upB)
                
        else:                                    
            for i in range (int(epoc)):        
                bmodel,upW,upB =md.backward(fmodel,weights,bias,outputTarget,learningRate)  
                fmodel= md.forward(bmodel,upW,upB)  
            
              
        printVal = []        
        for i in range(len(fmodel[0].norons)):
            printVal.append("i"+str(i+1)+"= "+str(fmodel[0].norons[i].value))
        for layer in range(1,len(fmodel)):
            for noron in range(len(fmodel[layer].norons)):
                printVal.append("h"+str(layer)+"-"+str(noron+1)+"= "+str(fmodel[layer].norons[noron].value))
        for noron in range(len(fmodel[len(fmodel)-1].norons)):
            printVal.append("o"+str(noron+1)+"= "+str(fmodel[len(fmodel)-1].norons[noron].value))

        main_window["-LISTBOX-"].update(values=printVal)        
        sse = md.sum_square_error(fmodel,outputTarget)        
        main_window["-SSE-"].update(value=sse)
        mse = md.mean_square_error(fmodel,outputTarget)
        main_window["-MSE-"].update(value=mse)
        rmse = md.root_mean_square_error(fmodel,outputTarget)
        main_window["-RMSE-"].update(value=rmse)                        
            
        main_window["-NEXT-"].update(disabled = True)              
        main_window["-BACK-"].update(disabled = False)        
        epoc = '0'   
    elif event == "-RESET-": 
        #TAM ANLAMIYLA ÇALIŞMIYOR
        main_window["-LISTBOX-"].update(values=[])        
        main_window["-INPUTDETAIL-"].update(disabled = False)
        main_window["-CREATEMODELVIEW-"].update(disabled = False)
        main_window["-HIDDENDETAIL-"].update(disabled = False)
        main_window["-HIDDENDETAILVALUE-"].update(visible = False) 
        main_window["-SECMODELVALUE-"].update(visible = False)
        main_window["-SELECTEDFILE-"].update("")
        main_window["-NUMBEROFINPUT-"].update("")
        main_window["-NUMBEROFOUT-"].update("") 
        main_window["-NUMBEROFLINES-"].update("")
        main_window['graph'].erase()       
        main_window["-SSE-"].update(value="")        
        main_window["-MSE-"].update(value="")        
        main_window["-RMSE-"].update(value="")
        epoc="0"
        # main_window.extend_layout(window['-SECMODELVALUE-'], select_detail_input(0,0,0)) 
        # main_window.extend_layout(window['-HIDDENDETAILVALUE-'], hidden_detail_input(0)) 