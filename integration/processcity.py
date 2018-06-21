#!C:\Miniconda3\envs\py36\python.exe
import os
import cgi
import numpy as np
import torch
import model
import pandas as pd
from flask import Flask, render_template
from sklearn import preprocessing

try:
	import torch 
except ImportError:
    server_msg("NO torch found")

#os.environ['city'] = 'beijing'
#city = os.getenv('city')
form=cgi.FieldStorage() 
city = form.getvalue("city")
num_steps = 24

def load():

	if(city=="montreal"):
		trained_model=model.LSTM(12,24,2)
		states=torch.load('montreal.pt')
		trained_model.load_state_dict(states)
		input_size =12
		pm25_index = -1
		
		#print(trained_model)
		#return trained_model
		
	if (city =="toronto"):
		trained_model=model.LSTM(13,24,2)
		states=torch.load('toronto.pt')
		trained_model.load_state_dict(states)
		input_size =13
		pm25_index = -1
		
		#return trained_model
		
	if (city =="ottawa"):
		trained_model=model.LSTM(13,24,2)
		states=torch.load('ottawa.pt')
		trained_model.load_state_dict(states)
		input_size =13
		pm25_index = -1
		return trained_model,input_size,pm25_index		
		
	if (city =="vancouver"):
		trained_model=model.LSTM(16,24,2)
		states=torch.load('vancouver.pt')
		trained_model.load_state_dict(states)
		input_size =16
		pm25_index = -3
		return trained_model,input_size,pm25_index
		#return trained_model
		
	if (city =="hamilton"):
		trained_model=model.LSTM(13,24,2)
		states=torch.load('hamilton.pt')
		trained_model.load_state_dict(states)
		input_size =13
		return trained_model,input_size,pm25_index
		
	if (city =="beijing"):
		trained_model=model.LSTM(1,24,2)
		states=torch.load('beijing.pt')
		trained_model.load_state_dict(states)
		input_size =1
		pm25_index = -1
		
		return trained_model,input_size,pm25_index
		#return trained_model
	
	if (city =="chengdu"):
		trained_model=model.LSTM(1,24,2)
		states=torch.load('chengdu.pt')
		trained_model.load_state_dict(states)
		input_size =1
		pm25_index = -1
		
		return trained_model,input_size,pm25_index
	
	if (city =="shanghai"):
		trained_model=model.LSTM(1,24,2)
		states=torch.load('shanghai.pt')
		trained_model.load_state_dict(states)
		input_size =1
		pm25_index = -1;
		
		return trained_model,input_size,pm25_index
		
	if (city =="shenyang"):
		trained_model=model.LSTM(1,24,2)
		states=torch.load('shenyang.pt')
		trained_model.load_state_dict(states)
		input_size =1
		pm25_index = -1
		
	return trained_model,input_size,pm25_index
		
#def compute(model):
	#print("ok")	

def server_msg(msg):
	print ("Content-Type: text/html")
	print("\r\n")
	print()
	print ("""
	<!DOCTYPE html>
	<html>
		Python error message: """)
	print(msg)
	print("""
	</body>
	</html>""")

#need a batch of 30 data
def predict_one_hour(model,x_valid_set,input_size,num_steps):
    predictions=torch.zeros(num_steps)
    for i, x in enumerate(x_valid_set):
        hidden=model.init_hidden(1)
        y_pred,_,_=model(x.contiguous().view(-1, 1, input_size),hidden)
#         predictions[i]=y_pred[:,-1]*(max_value[-1]-min_value[-1])+min_value[-1]
        predictions[i]=y_pred[:,-1]
    return predictions

def predict_full_sequence(model, x, input_size, num_steps,pm25_index,scaler):
	predictions = torch.zeros(num_steps)
	hidden = model.init_hidden(1)
#     y_pred, _, hidden = model(x.contiguous().view(-1, 1, input_size), hidden)
#     x = torch.cat((x, y_pred))
#     predictions[0] = y_pred[:,pm25_index]

	if city=='beijing' or city=='chengdu' or city=='shanghai' or city=='shenyang':
		x= x.unsqueeze(1)
	for i in range(0, num_steps):
		y_pred, _, hidden = model(x.contiguous().view(-1, 1, input_size), hidden)
		#       x = x*(min_max_valid[i,1]-min_max_valid[i,0])+min_max_valid[i,0]
		#print(y_pred.shape)
		#rint(x.shape)
		x = torch.cat((x, y_pred))
		x = x[1:]
		predictions[i] = torch.FloatTensor(scaler.inverse_transform(np.expand_dims(y_pred[0].data, axis=0))[:,-1])
#         predictions[i]=y_pred[:,pm25_index]
	return predictions

def read_data(city):
	dataset=pd.read_csv("C:/Users/Azu/Documents/GitHub/air-quality-ai/Data/canada/montreal/montreal2017.csv")
	if city=='montreal':
		dataset=pd.read_csv("C:/Users/Azu/Documents/GitHub/air-quality-ai/Data/canada/montreal/montreal2017.csv")
	
	elif city =='toronto':
		dataset=pd.read_csv("C:/Users/Azu/Documents/GitHub/air-quality-ai/Data/canada/toronto/toronto.csv")
	
	elif city == 'ottawa':
		dataset=pd.read_csv("C:/Users/Azu/Documents/GitHub/air-quality-ai/Data/canada/ottawa/ottawa.csv")
		
	elif city=='vancouver':
		dataset=pd.read_csv("C:/Users/Azu/Documents/GitHub/air-quality-ai/Data/canada/vancouver/vancouver.csv")
		dataset=dataset.iloc[:,2:].copy()
		
	elif city=='beijing' or city=='chengdu' or city=='shanghai' or city=='shenyang':
		dataset=pd.read_csv("C:/Users/Azu/Documents/GitHub/air-quality-ai/" +city + ".csv")
		dataset=dataset.iloc[:,-1].copy()
		
	dataset.replace("InVld",np.nan,inplace=True)
	dataset.replace("<Samp",np.nan,inplace=True)
	dataset.replace("Down",np.nan,inplace=True)
	dataset.replace("Calib",np.nan,inplace=True)
	dataset.replace("NoData",np.nan,inplace=True)
	dataset.replace(9999,np.nan,inplace=True)
	dataset.replace(-999,np.nan,inplace=True)
	dataset=dataset.astype(np.float32)
	
	for i in range(len(dataset)):
		if city=='beijing' or city=='chengdu' or city=='shanghai' or city=='shenyang':
			if np.isnan(dataset.iat[i]):
				if i==0:
					dataset.iloc[i]=dataset.iat[i+1]
				elif i==len(dataset)-1:
					dataset.iloc[i]=dataset.iat[i-1]
				else:
					dataset.iloc[i]=np.nanmean([dataset.iat[i-1],dataset.iat[i+1]])
				
		else:
			for j in range(len(dataset.columns)):
				if np.isnan(dataset.iat[i,j]):
					if i==0:
						dataset.iloc[i,j]=dataset.iat[i+1,j]
					elif i==len(dataset)-1:
						dataset.iloc[i,j]=dataset.iat[i-1,j]
					else:
						dataset.iloc[i,j]=np.nanmean([dataset.iat[i-1,j],dataset.iat[i+1,j]])
	return dataset	

def test_main():
	print ("Content-Type: text/html")
	print("\r\n")
	print()
	print ("""
	<!DOCTYPE html>
	<html>
		hello world from""")
	print(city)
	print("""
	</body>
	</html>""")
	
def main():
	model,input_size,pm25_index = load()
	dataset=read_data(city)
	scaler = preprocessing.MinMaxScaler()

	data_set=dataset.astype('float32')
	dataset=read_data(city)
	
	split=round(0.90*len(dataset))
	#data1=dataset.iloc[:split].copy()
	test_set=dataset.iloc[split:].copy()
	#scaler = preprocessing.MinMaxScaler() 
	if city=='beijing' or city=='chengdu' or city=='shanghai' or city=='shenyang':
		test_set=test_set.reshape(-1,1) #because 1D array 
	test_set1 = scaler.fit_transform(test_set)
	
	# test_set1=scaler.transform(test_set)
	seq_len=30 + 1
	#x=len(data1)-seq_len
	y=len(test_set)-seq_len
	# sequences = [np.asarray(data1[t:t+seq_len]) for t in range(x)]
	test_seq=[np.asarray(test_set[t:t+seq_len]) for t in range(y)]
	test_seq1=[np.asarray(test_set1[t:t+seq_len]) for t in range(y)]
	# seq=torch.FloatTensor(np.asarray(sequences))
	test_seq=torch.FloatTensor(np.asarray(test_seq))
	test_seq1=torch.FloatTensor(test_seq1)
	# split_row=round(0.80*seq.size(0))
	# x_train_set=seq[:split_row, :-1]
	# y_train_set=seq[:split_row, -1:]
	# x_valid_set=seq[split_row:, :-1]
	# y_valid_set=seq[split_row:, -1:]
	x_test_set=test_seq1[-1,:-1]
	y_test_set=test_seq[-1,-1:]
	
	#print(x_test_set.shape)
	x_test_set = x_test_set.squeeze(1)
	x = torch.FloatTensor(x_test_set)
	# size=x_valid_set.size(0)
	# print(y_valid_set)
	num_steps = 24 # Do not set to higher value due to memory constraint
	full_predictions = predict_full_sequence(model, x, input_size, num_steps,pm25_index,scaler)
	
	total = 0
	for i in range (0,23):
		total = total + full_predictions.data.numpy()[i]
	avg = round(total/24,2)
	
	#fig = plt.figure(figsize=(14, 7))
	# plt.plot(range(0, x.size(0)), x[:,pm25_index].data.numpy(), color="royalblue", label="Inputs")
	#y=y_test_set[index:index+num_steps,pm25_index]
	#plt.plot(y.data.numpy(),color='darkcyan')
	#plt.scatter(range(0,num_steps),full_predictions.data.numpy())
	print ("Content-Type: text/html")
	print("\r\n")
	print()
	print ("""
	<!DOCTYPE html>
	<html>
		hello world from""")
	print(city)
	print(full_predictions.data.numpy())
		
	print("""
	<head>
    <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
    <script type="text/javascript">
      google.charts.load('current', {'packages':['corechart']});
      google.charts.setOnLoadCallback(drawChart);

      function drawChart() {
		
        var data = google.visualization.arrayToDataTable([
          ['Hour', 'Prediction',  { role: 'style' }],
          [ 1,      """)
	print(full_predictions.data.numpy()[0])
	print(""", 'color: #13a9fa'],
          [ 2, """)
	print(full_predictions.data.numpy()[1])
	print(""", 'color: #f99608'],
          [ 3,""")    
	print(full_predictions.data.numpy()[2])
	print(""", 'color: #13a9fa'],
          [ 4,""")      
	print(full_predictions.data.numpy()[3])
	print(""", 'color: #f99608'],
          [ 5,""")
	print(full_predictions.data.numpy()[4])
	print(""", 'color: #13a9fa'],
          [ 6,""")
	print(full_predictions.data.numpy()[5])
	print(""", 'color: #f99608'], 
		  [7, """)
	print(full_predictions.data.numpy()[6])
	print(""", 'color: #13a9fa'], 
		  [8, """)
	print(full_predictions.data.numpy()[7])
	print(""", 'color: #f99608'], 
		  [9, """)
	print(full_predictions.data.numpy()[8])
	print(""", 'color: #13a9fa'], 
		  [10, """)
	print(full_predictions.data.numpy()[9])
	print(""", 'color: #f99608'], 
		  [11, """)
	print(full_predictions.data.numpy()[10])
	print(""", 'color: #13a9fa'], 
		  [12, """)
	print(full_predictions.data.numpy()[11])
	print(""", 'color: #f99608'], 
		  [13, """)
	print(full_predictions.data.numpy()[12])
	print(""", 'color: #13a9fa'], 
		  [14, """)
	print(full_predictions.data.numpy()[13])
	print(""", 'color: #f99608'], 
		  [15, """)
	print(full_predictions.data.numpy()[14])
	print(""", 'color: #13a9fa'], 
		  [16, """)
	print(full_predictions.data.numpy()[15])
	print(""", 'color: #f99608'], 
		  [17, """)
	print(full_predictions.data.numpy()[16])
	print(""", 'color: #13a9fa'], 
		  [18, """)
	print(full_predictions.data.numpy()[17])
	print(""", 'color: #f99608'], 
		  [19, """)
	print(full_predictions.data.numpy()[18])
	print(""", 'color: #13a9fa'], 
		  [20, """)
	print(full_predictions.data.numpy()[19])
	print(""", 'color: #f99608'], 
		  [21, """)
	print(full_predictions.data.numpy()[20])
	print(""", 'color: #13a9fa'], 
		  [22, """)
	print(full_predictions.data.numpy()[21])
	print(""", 'color: #f99608'], 
		  [23, """)
	print(full_predictions.data.numpy()[22])
	print(""", 'color: #13a9fa'], 
		  [24, """)
	print(full_predictions.data.numpy()[23])
	print(""", 'color: #f99608']
		]);

        var options = {
          title: 'Concetration per Hour',
          width: 1000,
		  height: 500,
		  bar: {groupWidth: "95%"},
          legend: 'none'
        };

        var chart = new google.visualization.ColumnChart(document.getElementById('chart_div'));

        chart.draw(data, options);
      }
    </script>
    </head>
    <body>
    <div id="chart_div" style="width: 1500px; height: 500px;"></div>
	The average concentration of PM 2.5 over the next 24 hours is: """)
	print(avg)
	print(""" ug/m^3. The corresponding AQI with this value is """)
	if avg <= 30.00:
		print("good")
	elif(avg > 30.00 and avg <= 60.00):
		print("satisfactory")
	elif (avg > 60.00 and avg <= 90.00):
		print("moderately polluted")
	elif (avg > 90.00 and avg <= 120.00):
		print("poor")
	elif (avg > 120.00 and avg <= 250.00):
		print("very poor")
	elif (avg > 250.00):
		print("severe")
	print(""".
	</body>
	</html>""")
	#return render_template("processed.html", city=city)

main()
