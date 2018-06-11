#processcity.py
import os
import cgi

#os.environ['city'] = 'beijing'
city = os.environ.get('city')

print (city)

def main():
	print ("Content-Type: text/html \r\n\r\n")
	print ("""
	<!DOCTYPE html>
	<html>
		hello world from""")
	print(city)
	print("""
	</body>
	</html>""")
	
	if(city == "beijing")
	
main()
