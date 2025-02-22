import streamlit as st
import pickle
import re
import nltk
from IPython.terminal.shortcuts.auto_suggest import resume_hinting

nltk.download('punkt')
nltk.download('stopwords')

#loading models
clf= pickle.load(open('clf.pkl','rb'))
tfidf=pickle.load(open('tfidf.pkl','rb'))

#to clean the new uploaded resume
def cleanResume(txt):
    cleanText = re.sub('http\S+\S',' ',txt)
    cleanText=re.sub('RT|cc',' ',cleanText)
    cleanText = re.sub('@\S+',' ',cleanText)
    cleanText = re.sub('#\S+',' ',cleanText)
    cleanText=re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""),' ',cleanText)
    cleanText=re.sub(r'[^\x00-\x7f]',' ',cleanText)
    cleanText = re.sub('\s+',' ',cleanText)

    return cleanText
# web app
# using streamlit library
def main():
    st.title("Resume Screening App")
    # to upload resume in a text or pdf format
    uploaded_file=st.file_uploader('Upload Resume',type=['txt','pdf'])

    # to check if upload file is not empty
    if uploaded_file is not None:
        try: # in this block it checks the format of the file whether it is in utf-8
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails , try decoding with 'latin-1',normally they are in latin
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = cleanResume(resume_text)
        # created the vectors using tfidf transform
        input_features = tfidf.transform([cleaned_resume])
        #the category of i'd is stored in this
        prediction_id = clf.predict(input_features)[0]
        st.write(prediction_id)

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            7: "Data Analyst",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            5: "Civil Engineer",
            0: "Advocate",
            21: "SAP Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",

        }
        category_name = category_mapping.get(prediction_id, 'Unknown')

        st.write("Predicted Category:",category_name)



# python_main
if __name__ == "__main__":
    main()