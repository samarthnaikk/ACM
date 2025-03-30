import google.generativeai as genai
def response(Colorcode):
	genai.configure(api_key="AIzaSyCJ-HFEUIWZe8drZl8KoeGSw8YeytDtNBA") 
	model = genai.GenerativeModel("gemini-1.5-flash")
	response = model.generate_content(f'''
		I will give you a list which has element's [r g b], you have to give me color name. I just want a name,  give the closest color from the given list:
	 Red, Blue, Yellow, Green, Black, White, Brown, Purple, Pink, Gray
		{Colorcode}
		''')
	return(response.text)

