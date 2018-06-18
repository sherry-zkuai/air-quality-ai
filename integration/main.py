from flask import Flask, render_template, flash, request, jsonify
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import requests
import json
import logging
import requests_toolbelt.adapters.appengine

import torch

def load(city):
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

def get_info():
	city = request.form['city']	
	res = load(city)
	
	return render_template("""
	<!DOCTYPE html>
	<html>
		hello world from"""
	+ city + """
	</body>
	</html>""")
	
def welcome():
	return render_template('select.html')

if __name__ == '__main__':
	app.run()