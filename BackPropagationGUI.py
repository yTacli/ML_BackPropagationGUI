# Yücel TACLI
import random
from pathlib import Path
import PySimpleGUI as sg
import numpy as np
import pandas as pd
import Models as md

tableValues = []   # tablo values
columnsName = [] # table values
seed = 3
momentum= 0.75
epoc = 0
progresMaxVal = 100

VIEWWEIGHT:int = 500
VIEWHEIGHT:int = 300
config = {
    "theme" : "DarkGrey9",   
}
def create_main_window():  
    sg.theme(config["theme"])   
    file_select_layout = [ 
        [sg.Button("Dosya Seç",key="-SELECTFILE-"),sg.Button("Dosya Detay",key="-FILEDETAIL-",disabled=True)],      
        [sg.Frame("Seçilen Dosya Bilgileri", [
            [sg.Text("Dosya Adı : "), sg.Text("",key="-SELECTEDFILE-"), sg.Push()],
            [sg.Text("Input Sayısı : "), sg.Text("",key="-NUMBEROFINPUT-"), sg.Push()],
            [sg.Text("Output Sayısı/Adları : "),sg.Text("",key="-NUMBEROFOUT-"), sg.Push()],
            [sg.Text("Satır Sayısı : "),sg.Text("",key="-NUMBEROFLINES-"), sg.Push()]
        ],expand_x=True)], 
        [sg.Frame("Model Bilgileri", [
            [sg.Text("Gizli Katman Sayısı :   "),
                sg.Spin([x for x in range(21)],initial_value=1,  key="-HIDDENN-", size=(6,1)),
                sg.Button("Detay Bilgileri",key="-HIDDENDETAIL-",disabled=True),
            ],                      
        ],key="-FILEINPUTDETAIL-",expand_x=True,expand_y=True)], 
    ]
    values_input_layout = [                                        
        [sg.Frame("Model Bilgileri", [
            [sg.Text("Giriş Katmanı Nöron Sayısı(max:20) :  "),
                sg.Spin([x for x in range(21)],initial_value=2,  key="-SECINPUTNUMBER-", size=(6,1))
            ],              
            [sg.Text("Çıkış Katmanı Nöron Sayısı(max:20) :   "),
                sg.Spin([x for x in range(21)],initial_value=2,  key="-SECOUTPUTNUMBER-", size=(6,1))
            ],
            [sg.Text("Gizli Katman Sayısı :   "),
                sg.Spin([x for x in range(21)],initial_value=1,  key="-SECHIDDENNUMBER-", size=(6,1))
            ],                    
            [sg.Button("Detay Bilgileri",key="-INPUTDETAIL-")] 
        ],key="-VALUEINPUTDETAIL-",expand_x=True,expand_y=True)]
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
            sg.Button("Modeli Oluştur", key="-CREATEMODELVIEW-",disabled=True),
            sg.Button("Başlangıç Değerleri", key="-INITIALIZE-",disabled=True),
            sg.Push(),
            sg.Button("RESET", key="-RESET-")
        ],
        [
            sg.StatusBar("Yücel TACLI - https://github.com/yTacli/ML_BackPropagationGUI.git"),
            sg.Push(),
            sg.ProgressBar(max_value=progresMaxVal, orientation='h', size=(50, 20), key="-PROGRESS-")
        ]
    ]
    window = sg.Window("Back Propataion GUI", layout=main_layout, finalize=True)
    window.set_min_size(window.size)
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
    biasDist = (hiddenX[-1]) + (((VIEWWEIGHT-50)-(hiddenX[-1]))/2)
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
    for i in range(hidden[-1]):
        for j in range(output):
            g.draw_line((hiddenX[-1]+15,hiddenY[-1][i]),(VIEWWEIGHT-65, outY[j]),width=2)  
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
        [sg.Frame("Model Detay Bilgileri", 
        [
            [sg.Column(inp_hidden),
            sg.Column(inp_col),            
            sg.Column(out_col)],
            [   
                sg.Text("Öğrenme Oranı : "),
                sg.Spin([x for x in np.arange(0.0, 1.01, 0.01)], initial_value=0.05,  key="-LEARNRATE-", size=(6,1))
            ],
            [
                sg.Text("Aktivasyon Fonksiyonu : "),
                sg.Combo(['Sigmoid','ReLU','Threshold'],'Sigmoid', key='-ACTIVATION-')
            ],
            [
                sg.Text("Threshold : "),
                sg.Spin([x for x in np.arange(0.0, 1.01, 20.0)],initial_value=0.0,  key="-THRESHOLD-", size=(6,1))                    
            ]
        ], expand_x=True, expand_y=True, key="-SECMODELVALUE-")]
    ]
    return layout

def hidden_detail_input(hiddenLayerNumber):
    # 3'lü gruplarda sg.column yapılabilir.
    noron_layout =  [
        [
            sg.Text(f"Hidden-{x}-Noron Sayısı : "),
            sg.Input(default_text=1, key=f"-fh{x}-", size=(10,1),justification="center")       
        ] for x in range(1,hiddenLayerNumber+1)
    ]
    layout = [
        [sg.Column(noron_layout)],
        [sg.Frame("Model Detay Bilgileri", [
            [   
                sg.Text("Öğrenme Oranı : "),
                sg.Spin([x for x in np.arange(0.0, 1.01, 0.01)], initial_value=0.05,  key="-LEARNRATE-", size=(6,1))
            ],
            [
                sg.Text("Aktivasyon Fonksiyonu : "),
                sg.Combo(['Sigmoid','ReLU','Threshold'],'Sigmoid', key='-ACTIVATION-')
            ],
            [
                sg.Text("Threshold : "),
                sg.Spin([x for x in np.arange(0.0, 1.01, 20.0)],initial_value=0.0,  key="-THRESHOLD-", size=(6,1))                                    ]
        ], expand_x=True, expand_y=True, key="-HIDDENDETAILVALUE-")],                   
    ] 
    return layout

def file_row_select(dataFrame,rowsNumber,inputColumns,ouputColumn):
    selectRow = list(dataFrame.loc[rowsNumber])
    selectRowInput = [] 
    for col in inputColumns:
        if selectRow[col] != "?":
            selectRowInput.append(float(selectRow[col])) # verilerde string varsa float()
        else:
            selectRowInput.append(float(0.0))

    # 2 output için
    if selectRow[ouputColumn] == 0:
        out2 = 1
    else:
        out2 = 0

    selectRowOutput = []
    selectRowOutput.append(selectRow[-1])
    selectRowOutput.append(out2)

    return selectRowInput,selectRowOutput

def create_detay_window(dataFrame):    
    sg.theme(config["theme"]) 
    try:
        columnsName = list(dataFrame.columns)          
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

        select_layout = [
            [
                sg.Text(f"{column} : "),
                sg.Combo(["ID","INPUT","OUTPUT"],"INPUT",auto_size_text=True,key=f"-{column}-")
            ] for column in columnsName
        ]               
        main_layout= [
            [sg.Push(), sg.Text("VERİ SEÇİMİ"),sg.Push()],
            [sg.Table(values=tableValues, headings=columnsName, expand_x=True, expand_y=True,auto_size_columns=True,)],
            [sg.Frame("Kolonlar",[
                [sg.Column(select_layout)]
            ]
            )],           
            [sg.Button("Devam",key="-SELECTCONTINUE-")]
        ]        
        window = sg.Window("VERİ SEÇİM", layout=main_layout, finalize=True)  
        return window,columnsName
    except:
        pass       


main_window = create_main_window()
while 1:
    window, event, values = sg.read_all_windows() 
    selectedTab = main_window["-TAB-"].get()

    if event == sg.WIN_CLOSED or event == 'Exit':       
        window.close()
        detail_window = None
        if window == detail_window:       
            detail_window = None
            main_window.refresh()
        elif window == main_window:     
            break
    elif event == "-INPUTDETAIL-":         
        if selectedTab == "Veri Giriş":
            secInputNumber = int(main_window['-SECINPUTNUMBER-'].get())            
            secOutputNumber = int(main_window['-SECOUTPUTNUMBER-'].get())            
            secHiddenLayerNumber = int(main_window['-SECHIDDENNUMBER-'].get())
            if secInputNumber <= 0 or secHiddenLayerNumber <= 0 or secOutputNumber <=0:
                sg.popup("Değerler SIFIR veya sıfırdan küçük olamaz olamaz")
            else:
                main_window.extend_layout(window["-VALUEINPUTDETAIL-"], select_detail_input(secInputNumber,secHiddenLayerNumber,secOutputNumber))                  
            main_window["-INPUTDETAIL-"].update(disabled = True)           
            main_window["-INPUTDETAIL-"].update(disabled = True)
            main_window["-CREATEMODELVIEW-"].update(disabled = False) 
    elif event == "-HIDDENDETAIL-":
        if selectedTab == "Dosya Seçim":
            fileHiddenLayerNumber = int(main_window["-HIDDENN-"].get())
            main_window.extend_layout(window['-FILEINPUTDETAIL-'], hidden_detail_input(fileHiddenLayerNumber))
            main_window["-HIDDENDETAIL-"].update(disabled = True)           
            main_window["-CREATEMODELVIEW-"].update(disabled = False)
    elif event == "-FILEDETAIL-":
        detail_window, columnsName = create_detay_window(df)
    elif event == "-SELECTFILE-":
        folder_or_file = sg.popup_get_file('Dosya Seçiminiz','Dosya Seçimi', keep_on_top=True,file_types=(('TXT Files', '*.txt'),))
        try:            
            df = pd.read_table(folder_or_file, delimiter="\t")    
            main_window["-FILEDETAIL-"].update(disabled = False)                                          
            main_window["-SELECTFILE-"].update(disabled = True)
        except:
            pass   
    elif event == "-SELECTCONTINUE-":                   
        rowsNumber = df.shape[0] - 1    # satır sayısı (başlıkları sayma) 
        fileInputNumber = 0
        inputColumns = []      
        outputColumnsNumber = None
        for i in range(len(columnsName)):
            for key in values:
                if key == "-" + str(columnsName[i]) + "-":
                    if values[key] == "INPUT":
                        fileInputNumber += 1
                        inputColumns.append(i)
                    if values[key] == "OUTPUT":                        
                        outputColumnsNumber = i         
        if  fileInputNumber != 0 and outputColumnsNumber != None:            
            # output gruplama
            outGrup = df.groupby(str(columnsName[outputColumnsNumber]))       
            fileOutputNumber = len(outGrup.groups)  # class sayısı  
            outstr = outGrup.groups.keys()  # class isimleri           
            outputString = str(fileOutputNumber) + " / " + str(list(outstr))

            main_window["-SELECTEDFILE-"].update(Path(folder_or_file).stem) 
            main_window["-NUMBEROFINPUT-"].update(fileInputNumber)
            main_window["-NUMBEROFOUT-"].update(outputString) 
            main_window["-NUMBEROFLINES-"].update(rowsNumber)
            main_window["-CREATEMODELVIEW-"].update(disabled = True)            
            main_window["-FILEDETAIL-"].update(disabled = True)
            main_window["-HIDDENDETAIL-"].update(disabled = False)
            detail_window.close()
        else:
            sg.Popup("Input veya Output için Seçim Yapmadınız!!!")
    elif event == "-CREATEMODELVIEW-":        
        activationFunction = values['-ACTIVATION-']         
        if activationFunction == "":            
            sg.popup("Activasyon Fonksiyonu Seçmediniz!!")
            main_window['-ACTIVATION-'].set_focus=True
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
            sonsutun = tempdatas.columns[outputColumnsNumber]
            for i in range(fileOutputNumber):
                for j in range(rowsNumber):
                    if tempdatas.iloc[j][fileInputNumber+1] == list(outstr)[i]:
                        tempdatas.at[j,sonsutun] = i

            
            rndRow = random.randint(1,rowsNumber)

            fileInput, outputTarget =  file_row_select(tempdatas,rndRow,inputColumns,outputColumnsNumber)     # ilk sutun
        
        main_window["-CREATEMODELVIEW-"].update(disabled = True) 
        main_window["-INITIALIZE-"].update(disabled = False)       
    elif event == "-INITIALIZE-":           
        threshold = float(main_window["-THRESHOLD-"].get())
        learningRate = float(main_window['-LEARNRATE-'].get())     

        if selectedTab == "Veri Giriş":
            model,weights,bias,prewWeightsDelta,prewBiasDelta = md.create_model_base(secInputNumber,inputValues,secHiddenLayerNumber,secHiddenList,secOutputNumber,seed,activationFunction,threshold)            
        else:
            model,weights,bias,prewWeightsDelta,prewBiasDelta = md.create_model_base(fileInputNumber,fileInput,fileHiddenLayerNumber,fileHiddenList,fileOutputNumber,seed,activationFunction,threshold)        

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
        model = md.forward(model,weights,bias)    

        printVal = []        
        for i in range(len(model[0].norons)):
            printVal.append("i"+str(i+1)+"= "+str(model[0].norons[i].value))
        for layer in range(1,len(model)-1):
            for noron in range(len(model[layer].norons)):
                printVal.append("h"+str(layer)+"-"+str(noron+1)+"= "+str(model[layer].norons[noron].value))
        for noron in range(len(model[-1].norons)):
            printVal.append("o"+str(noron+1)+"= "+str(model[-1].norons[noron].value))

        main_window["-LISTBOX-"].update(values=printVal)        
        sse = md.sum_square_error(model,outputTarget)        
        main_window["-SSE-"].update(value=sse)
        mse = md.mean_square_error(model,outputTarget)
        main_window["-MSE-"].update(value=mse)
        rmse = md.root_mean_square_error(model,outputTarget)
        main_window["-RMSE-"].update(value=rmse)
        
        main_window["-NEXT-"].update(disabled = True)              
        main_window["-BACK-"].update(disabled = False)        
    elif event == "-BACK-":    
        model,upW,upB,prewWDelta,prewBDelta = md.backward(model,weights,bias,outputTarget,learningRate,momentum,prewWeightsDelta,prewBiasDelta)
        printVal = []
        for layer in range(len(weights)):
            for noron in range(len(weights[layer])):
                for nextNoron in range(len(weights[layer][noron])): # next noron
                    printVal.append("old_w"+str(layer+1)+"_"+str(noron+1)+"-"+str(nextNoron+1)+"= "+str(weights[layer][noron][nextNoron]))
                    printVal.append("new_w"+str(layer+1)+"_"+str(noron+1)+"-"+str(nextNoron+1)+"= "+str(upW[layer][noron][nextNoron]))
        printVal.append("BIAS")        
        for i in range(len(bias)): 
            for j in range(len(bias[i])):
                printVal.append("old_b"+str(i+1)+"-"+str(j+1)+"= "+str(bias[i][j]))
                printVal.append("new_b"+str(i+1)+"-"+str(j+1)+"= "+str(upB[i][j]))

        prewWeightsDelta = prewWDelta   
        prewBiasDelta = prewBDelta 
        model = model
        weights = upW
        bias = upB

        main_window["-LISTBOX-"].update(values=printVal)
        main_window["-NEXT-"].update(disabled = False)              
        main_window["-BACK-"].update(disabled = True) 
        main_window["-PLAY-"].update(disabled = False) 
    elif event == "-PLAY-":    
        bar = main_window["-PROGRESS-"]
        bar.update_bar(0)
        maxEpoc = sg.popup_get_text('epoc', "EPOC SAYISI",'1')
        barUP = float(progresMaxVal) / float(maxEpoc)
        progress = 0
        rnd = random.Random(seed)       
        if selectedTab == "Dosya Seçim":  
            rowIndexs = np.arange(rowsNumber)
            epoc = 0
            # İlklendirildiği için aynı sonucu verecektir. 
            # Döngü Backward ile başlayacağı için ilk değer olmazsa hata verecektir.        
            while epoc < int(maxEpoc):                                         
                rnd.shuffle(rowIndexs)
                for ri in range(rowsNumber):
                    fileInput, outputTarget =  file_row_select(tempdatas,ri,inputColumns,outputColumnsNumber)    
                    for i in range(len(model[0].norons)):
                        model[0].norons[i].value = fileInput[i]    # input
                    for j in range(len(model[-1].norons)):
                        model[-1].norons[j].value = outputTarget[j]  # output

                    model= md.forward(model,weights,bias)
                    model,upW,upB,prewWDelta,prewBDelta = md.backward(model,weights,bias,outputTarget,learningRate,momentum,prewWeightsDelta,prewBiasDelta) 
                    weights = upW
                    bias = upB
                epoc += 1  
                progress += barUP
                bar.update_bar(progress) 
                if progress >= progresMaxVal:
                        sg.popup("Eğitim Sona Erdi...")
        else:
             while epoc < int(maxEpoc): 
                model= md.forward(model,weights,bias)                                   
                model,upW,upB,prewWDelta,prewBDelta = md.backward(model,weights,bias,outputTarget,learningRate,momentum,prewWeightsDelta,prewBiasDelta)
                weights = upW
                bias = upB 
                epoc += 1  
                progress += barUP
                bar.update_bar(progress) 
                if progress >= progresMaxVal:
                        sg.popup("Eğitim Sona Erdi...")
        printVal = []   
        model= md.forward(model,upW,upB)

        for i in range(len(model[0].norons)):
            printVal.append("i"+str(i+1)+"= "+str(model[0].norons[i].value))
        for layer in range(1,len(model)-1):
            for noron in range(len(model[layer].norons)):
                printVal.append("h"+str(layer)+"-"+str(noron+1)+"= "+str(model[layer].norons[noron].value))
        for noron in range(len(model[-1].norons)):
            printVal.append("o"+str(noron+1)+"= "+str(model[-1].norons[noron].value))

        main_window["-LISTBOX-"].update(values=printVal)        
        sse = md.sum_square_error(model,outputTarget)        
        main_window["-SSE-"].update(value=sse)
        mse = md.mean_square_error(model,outputTarget)
        main_window["-MSE-"].update(value=mse)
        rmse = md.root_mean_square_error(model,outputTarget)
        main_window["-RMSE-"].update(value=rmse)                        
                
        main_window["-NEXT-"].update(disabled = True)              
        main_window["-BACK-"].update(disabled = False)        
        epoc = '0'   
    elif event == "-RESET-":        
        window.close()       
        main_window = create_main_window()