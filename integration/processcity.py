#!C:\Python\python.exe
import numpy
import os
import cgi

#os.environ['city'] = 'beijing'
#city = os.getenv('city')
form=cgi.FieldStorage() 
city = form.getvalue("city")

def load():
	if(city=="montreal"):
		trained_model=model.LSTM(12,24,2)
		states=torch.load('montreal.pt')
		trained_model.load_state_dict(states)
		
		return trained_model
		
	if (city =="toronto"):
		trained_model=model.LSTM(13,24,2)
		states=torch.load('toronto.pt')
		trained_model.load_state_dict(states)
		
		return trained_model
		
	if (city =="ottawa"):
		trained_model=model.LSTM(13,24,2)
		states=torch.load('ottawa.pt')
		trained_model.load_state_dict(states)
		
		return trained_model
		
	if (city =="vancouver"):
		trained_model=model.LSTM(16,24,2)
		states=torch.load('ottawa.pt')
		trained_model.load_state_dict(states)
		
		return trained_model
		
#def compute(model):
	#print("ok")
	
def main():
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

main()
