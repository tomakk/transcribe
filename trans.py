import streamlit as st
import openai
import os
import whisper
from pydub import AudioSegment
import pandas as pd
import ast
from io import BytesIO

openai.api_key = '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'

def convert_mp3_to_wav(mp3_file_path):
    audio = AudioSegment.from_mp3(mp3_file_path)
    wav_file_path = mp3_file_path.replace(".mp3", ".wav")
    audio.export(wav_file_path, format="wav")
    return wav_file_path

def transcribe_audio(file_path, model):
    result = model.transcribe(file_path, language="de")
    return result["text"]

def parse_transcription(transcribed_text):
    sections = {
        "Einleitung": "",
        "Fragestellung": "",
        "Anamnese/Klinische Angaben": "",
        "Technik": "",
        "Befund": "",
        "Beurteilung": "",
        "Diagnose": "",
        "Anhang": "",
        "Grußformel": ""
    }

    base_instruction = f'''Ignorieren Sie alle vorherigen Anweisungen.
        Sie sind ein deutscher Radiologietechniker und haben die Aufgabe, den deutschen Radiologiebericht {transcribed_text} in eine Zielberichtsstruktur mit Abschnitten einzufügen. 
        Ihr Ziel ist es, genaue, vollständige und qualitativ hochwertige Berichte für den Radiologietechniker für jede Dialoglinie sicherzustellen. 
        Die Eingabe besteht aus einem Radiologiebericht in deutscher Sprache, der Text wie XXXX enthalten kann, eine Ansammlung von X, die ignoriert/bereinigt werden muss. 
        Ihre Aufgabe besteht darin, zunächst alle Fehler, die Sie in den Sätzen finden, anhand ihres Kontexts zu korrigieren und sie dann entsprechend dem radiologischen Kontext den Zielabschnitten zuzuordnen.
        Der Benutzer stellt einen Teil der Zeilen auf Deutsch bereit. Sie sollten mit einer genauen, prägnanten und vollständigen Zuordnung des Berichtstexts zu den Abschnitten antworten: {sections} . 
        Der gesamte Eingabetext muss einem Abschnitt zugeordnet werden, der kontextuell am relevantesten ist.  
        Ihre Antwort wird von einem automatisierten System verarbeitet, daher ist es unbedingt erforderlich, dass Sie das erforderliche Ausgabeformat einhalten.'''

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": base_instruction}
        ],
        max_tokens=2200,
        temperature=0.2
    )
    
    completion_text = response['choices'][0]['message']['content'].strip()
    print("Raw completion text:")
    print(completion_text)
    
    try:
        sections = ast.literal_eval(completion_text)
    except (ValueError, SyntaxError) as e:
        print(f"Error decoding response: {e}")
        return sections
    
    for section in sections:
        sections[section] = sections[section].strip()
    
    return sections

def process_audio_file(audio_file):
    data = []
    error_files = []
    
    try:
        model = whisper.load_model("small")
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return {}

    with open("temp.mp3", "wb") as f:
        f.write(audio_file.getbuffer())
    
    wav_file = convert_mp3_to_wav("temp.mp3")
    try:
        transcription = transcribe_audio(wav_file, model)
        structured_data = parse_transcription(transcription)
        data.append(structured_data)
    except Exception as e:
        st.error(f"Could not process file: {e}")
        error_files.append(audio_file.name)
    finally:
        os.remove("temp.mp3")
        if os.path.exists(wav_file):
            os.remove(wav_file)
    
    if error_files:
        st.warning("The following files could not be processed:")
        for error_file in error_files:
            st.warning(error_file)
    
    return data[0] if data else {}

st.title("Radiology Report Transcription and Structuring")

uploaded_file = st.file_uploader("Upload an MP3 file", type="mp3")

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')

    with st.spinner('Processing...'):
        structured_data = process_audio_file(uploaded_file)
    
    st.success('Processing complete!')

    if structured_data:
        st.header("Transcribed and Structured Report")
        for section, content in structured_data.items():
            st.subheader(section)
            st.write(content)