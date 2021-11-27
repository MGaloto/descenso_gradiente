

# Librerias

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import time

#%%

# Grafico Inflacion


df = pd.read_excel('argentina.xlsx')

# Grafico de barras con los 10 paises de mayor PBI

df_inflacion = df[df['inflacion']<200]


pio.renderers.default='svg'

fig = px.bar(df_inflacion, y='inflacion', x='año',  text='inflacion')

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.update_layout(title_text="Inflacion Anual Argentina 1961 - 2020 con tasas menores al 200%")
fig.show()


#%%


pio.renderers.default='svg'

fig = px.bar(df, y='resultado_economico', x='año',  text='resultado_economico')

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.update_layout(title_text="Resultado Economico Anual Argentina 1961 - 2020")
fig.show()




#%%

# Series temporales de recaudacion y presion tributaria

from plotly.subplots import make_subplots

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Scatter(x=df['año'], y=df['recaudacion_en_mill'], 
                    mode='lines+markers',
                    name='recaudacion_en_mill'),secondary_y=False)
fig.add_trace(go.Scatter(x=df['año'], y=df['gastopbi'], 
                    mode='lines+markers',
                    name='gastopbi'),secondary_y=True)
fig.update_layout(title_text="Recaudacion Base 2015 y Gasto/PBI")
fig.show()





#%%

# Regresion Polinomica


x = df['gastopbi']
y = df['recaudacion_en_mill']

f = np.polyfit(x,y, 3)

p = np.poly1d(f)


print(p)



#%%

import matplotlib.pyplot as plt 

def PlotPolly(model, independent_variable, dependent_variable, Name):
  x_new = np.linspace(x.min(),x.max(), 100)
  y_new = model(x_new)
  sns.scatterplot(independent_variable,dependent_variable,c=df.año, cmap='Reds')
  sns.lineplot(x_new, y_new)
  plt.title('Recaudacion Real y Presion Tributaria')
  plt.xlabel('Presion Tributaria')
  plt.ylabel('Recaudacion Real')
  plt.show()



PlotPolly(p, x, y, 'gastopbi')





#%%


function = lambda x: (-0.2459 * (x ** 3)) + (20.48 *(x ** 2)) - (500.2 * x) + 4134

x = np.linspace(x.min(), x.max(), 60)



 
#%%


nuevax = []
nuevay = []



def derivada(x):
    x_deriv = (-0.7377 * (x ** 2)) + (40.96 *x) - 500.2 
    return x_deriv


#%%


# Algoritmo

def gradiente(x_new, x_prev, precision, leanrn_rate):
    x_list =  [x_new]
    y_list =  [function(x_new)]

    while abs(x_new - x_prev) > precision:
        x_prev = x_new
        d_x =  derivada(x_prev)
        x_new = x_prev - (leanrn_rate * d_x)
        x_list.append(x_new)
        y_list.append(function(x_new))
        nuevax.append(x_new)
        nuevay.append(function(x_new))
        plt.scatter(x_list,y_list,c="b")
        plt.plot(x_list,y_list,c="r")
        plt.plot(x,function(x), c="r")
        plt.title("Curva de Laffer")
        plt.xlabel('Presion Tributaria')
        plt.ylabel('Recaudacion Real')
        plt.show()

        time.sleep(1)
        print('Iteracion numero: ' ,str(len(nuevax)) ,
              '\nPresion tributaria de:' ,str(x_new)[0:6],
              '\n')
    
    print ("El minimo de recaudacion se encuentra cuando la presion tributaria es de: "+ str(x_new)[0:6])
    print ("Numero de iteraciones: " + str(len(nuevax)), '\n')
    time.sleep(6)
    
gradiente(x.min(), 0, 0.01, 0.03)
    


def gradiente(x_new, x_prev, precision, leanrn_rate):
    x_list =  [x_new]
    y_list =  [function(x_new)] 

    while abs(x_new - x_prev) > precision:
        x_prev = x_new
        d_x =  derivada(x_prev)
        x_new = x_prev + (leanrn_rate * d_x)
        x_list.append(x_new)
        y_list.append(function(x_new))
        nuevax.append(x_new)
        nuevay.append(function(x_new))
        plt.scatter(x_list,y_list,c="b")
        plt.plot(x_list,y_list,c="r")
        plt.plot(x,function(x), c="r")
        plt.title("Curva de Laffer")
        plt.xlabel('Presion Tributaria')
        plt.ylabel('Recaudacion Real')
        plt.show()

        time.sleep(1)
        print('Iteracion numero: ' ,str(len(nuevax)) ,
              '\nPresion tributaria de:' ,str(x_new)[0:6],
              '\n')
    
    print ("El maximo de recaudacion se encuentra cuando la presion tributaria es de: "+ str(x_new)[0:6])
    print ("Numero de iteraciones: " + str(len(nuevax)), '\n')

    
gradiente(nuevax[-1] + 0.05 ,0 , 0.01, 0.03)



#%%
    

