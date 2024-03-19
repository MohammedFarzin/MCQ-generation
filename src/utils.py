import os
import PyPDF2
import json
import traceback


def read_file(file):
    if file.name.endswith('.pdf'):
        try:
            pdf_reader = PyPDF2.PdfFileReader(file)
            text = ""
            for page in pdf_reader.numPages:
                text += page.extract_text()
            return text
        except Exception as e:
            raise e("Error reading pdf file")
    
    elif file.name.endswith('.txt'):
        return file.read().decode('utf-8')
    
    else:
        raise Exception("File type not supported")


def get_table_data(quiz_str):
    try:
        quiz=json.loads(response.get("quiz"))     # loading response into json object
        quiz_table_data = []

        # iterating over the dictionary items
        for key, value in quiz.items():
            mcq = value["mcq"]
            options = " | ".join(
                [
                    f"{option}: {option_value}"
                    for option, option_value in value["options"].items()
                    ]
                )
            correct = value["correct"]
            quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})

        quiz=pd.DataFrame(quiz_table_data) 
        return quiz

    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        return False
