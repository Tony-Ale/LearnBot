import re
import json
from typing import List
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from async_youtube_transcript_api import YouTubeTranscriptApi
from .data_store import Data_Store
import math
import asyncio

def invoke_llm(sentences:str)->str:
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="""You're an expert at crafting multiple choice questions. Your task is to create multiple choice questions based on the content and the format given.
                        WARNING: You must follow the format below strictly, never in any case must you deviate from the format given.
                        WARNING: You must follow the format for each question
                        WARNING: You must not add any additional text or character or remove any text, follow my format strictly
                        WARNING: As shown in the format, two new line characters should be used to seperate questions
                        Format: \n"
                        Question: <question_text>\n
                        (A) <option_A>\n
                        (B) <option_B>\n
                        (C) <option_C>\n
                        (D) <option_D>\n
                        Answer: <correct_option>\n\n
                    """
                ),
        ("user", "using the format given to you, Generate multiple multiple-choice questions based on the content below:\n\n{sentences}")
    ])
    
    llm = Data_Store.llm
    prompt = prompt_template.format_messages(sentences=sentences)
    output = llm.invoke(prompt)


    mcq = output.content

    return mcq

async def async_invoke_llm(transcripts:list)->list:
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="""You're an expert at crafting multiple choice questions. Your task is to create multiple choice questions based on the content and the format given.
                        WARNING: You must follow the format below strictly, never in any case must you deviate from the format given.
                        WARNING: You must follow the format for each question
                        WARNING: You must not add any additional text or character or remove any text, follow my format strictly
                        WARNING: As shown in the format, two new line characters should be used to seperate questions
                        Format: \n"
                        Question: <question_text>\n
                        (A) <option_A>\n
                        (B) <option_B>\n
                        (C) <option_C>\n
                        (D) <option_D>\n
                        Answer: <correct_option>\n\n
                    """
                ),
        ("user", "using the format given to you, Generate multiple multiple-choice questions based on the content below:\n\n{transcript}")
    ])
    
    llm = Data_Store.llm

    prompts = []
    for transcript in transcripts:
        prompt = prompt_template.format_messages(transcript=transcript)
        prompts.append(prompt)

    output = await llm.abatch(prompts)

    mcqs = []
    for response in output:
        mcqs.append(response.content)  

    return mcqs

def filter_transcript(transcript:list[dict], max_num_words=3000)->List[str]:
    """
    if the number of words of the transcript is greater than the specified max number of words this function will 
    split the transcript into 3; if max_num_words = 3000, the final filtered transcript will contain the first 1000 words,
    the first thousand word starting from the middle, and the last one thousand words without overlapping.
    """
    data = []

    if transcript is None:
        return data

    # get the total number of words 
    total_num_of_words = 0
    num_lines = 0
    for t in transcript:
        text:str = t['text']
        total_num_of_words += len(text.split())
        num_lines += 1

    words_per_line = total_num_of_words/num_lines

    if total_num_of_words > max_num_words:
        # get the first one thousand words that is if max_num_words == 3000
        n = max_num_words/3
        num_lines_for_n_number_of_words = math.floor(n/words_per_line)

        # get the first one thousand words, starting from the middle of the transcript
        middle_val = total_num_of_words/2
        line_num_for_start_of_text_from_middle = math.floor(middle_val/words_per_line)
        end_of_middle_transcript = line_num_for_start_of_text_from_middle + num_lines_for_n_number_of_words

        # get the last one thousand words
        line_num_for_start_of_last_thousand_words = math.floor((total_num_of_words - n)/words_per_line)

        # in order to prevent overlap
        if line_num_for_start_of_last_thousand_words <= end_of_middle_transcript:
            line_num_for_start_of_last_thousand_words = end_of_middle_transcript + 1

        for idx, t in enumerate(transcript):
            if idx < num_lines_for_n_number_of_words:
                data.append(t['text'])
            elif idx >= line_num_for_start_of_text_from_middle and idx < end_of_middle_transcript:
                data.append(t['text'])
            elif idx >= line_num_for_start_of_last_thousand_words:
                data.append(t['text'])
    else:
        data = [t["text"] for t in transcript]
    
    return data
        
def extract_needed_transcripts(video_ids:list[str], transcripts_data:dict[str, list[str]]):
    extracted_transcripts = {}
    for video_id in video_ids:
        extracted_transcripts[video_id] = transcripts_data[video_id]
    return extracted_transcripts

def is_extracted_transcripts_none(extracted_transcript:dict[str, list|None]):
    # Check if all values are None
    all_none = all(value is None for value in extracted_transcript.values())

    return all_none

# Function to retrieve transcript data
def get_transcript_data(video_ids:list[str], 
                    video_complete: bool = True, 
                    watch_duration: float = None, 
                    *, 
                    transcripts_data:dict[str, list[dict]]=None):
    
    data_dict:dict[str, list[str]] = {}
    num_workers = min(5, math.ceil(len(video_ids)/2))

    if YouTubeTranscriptApi.session_created:
        YouTubeTranscriptApi.session_created = False

    if transcripts_data is None:
        response = YouTubeTranscriptApi.get_transcripts(video_ids, 
                                                        num_workers=num_workers,
                                                        languages=['en', 'en-US'],
                                                        continue_after_error=True)
        transcripts, unretrievable_ids = response
    else:
        transcripts = extract_needed_transcripts(video_ids=video_ids,
                                                 transcripts_data=transcripts_data)
        if is_extracted_transcripts_none(transcripts):
            response = YouTubeTranscriptApi.get_transcripts(video_ids, 
                                                            num_workers=num_workers,
                                                            languages=['en', 'en-US'],
                                                            continue_after_error=True)
            transcripts, unretrievable_ids = response

    for video_id, transcript in transcripts.items():
        if video_complete:
            data = filter_transcript(transcript)
            #data = [t["text"] for t in transcript]
        else:
            vid_duration = transcript[-1]["start"] + transcript[-1]["duration"]
            use_transcript = [t for t in transcript if t["start"] <= watch_duration]
            data = [t["text"] for t in use_transcript]

        data = [re.sub(r"[^a-zA-Z0-9-ışğöüçiIŞĞÖÜÇİ ]", "", line) for line in data]
        data_dict[video_id] = data
    return data_dict

# Function to extract MCQs from generated text
def extract_mcqs(data:str, map_answer_to_text=False)->List[str]:
    """
    Extracts multiple-choice questions, their options, and answers from a formatted string.

    This function addresses the issue of capturing questions and options that may span multiple lines.
    It uses specific markers to ensure complete extraction:
    - The end of the question is identified by the occurrence of the option label (A).
    - The options are captured using the subsequent letters of the alphabet (B, C, D) as delimiters.
    """
    answer_to_text_index_map = {'A':0, 'B':1, 'C':2, 'D':3}

    # Regex patterns
    question_pattern = r"Question:(.*?)\(A\)"
    option_a_pattern = r"\(A\)\s*(.*?)\s*\(B\)"
    option_b_pattern = r"\(B\)\s*(.*?)\s*\(C\)"
    option_c_pattern = r"\(C\)\s*(.*?)\s*\(D\)"
    option_d_pattern = r"\(D\)\s*(.*?)\s*Answer:"
    answer_pattern = r"Answer:\s*([A-D])"

    # Find all positions of 'Question:'
    question_positions = [(m.start(), m.end()) for m in re.finditer(r"Question:", data)]

    # Iterate over all question positions and extract the question, options, and answer
    extracted_data = []
    for i, (start_pos, end_pos) in enumerate(question_positions):
        # Find the start and end of each question block
        next_start_pos = question_positions[i + 1][0] if i + 1 < len(question_positions) else len(data)

        # Extract the question block
        question_block = data[start_pos:next_start_pos].strip()

        # Extract the question text (from 'Question:' to '(A)')
        question_match = re.search(question_pattern, question_block, re.DOTALL)
        if question_match:
            question_body = question_match.group(1).strip()

            # Find the options using individual patterns for A, B, C, D
            option_a = re.search(option_a_pattern, question_block, re.DOTALL)
            option_b = re.search(option_b_pattern, question_block, re.DOTALL)
            option_c = re.search(option_c_pattern, question_block, re.DOTALL)
            option_d = re.search(option_d_pattern, question_block, re.DOTALL)

            # Find the answer
            answer_match = re.search(answer_pattern, question_block)


            # Check if all options and answer are found
            if option_a and option_b and option_c and option_d and answer_match:
                options = [
                    option_a.group(1).strip(),
                    option_b.group(1).strip(),
                    option_c.group(1).strip(),
                    option_d.group(1).strip()
                ]
                answer = answer_match.group(1).strip()
                if map_answer_to_text:
                    answer_index = answer_to_text_index_map[answer.upper()]
                    answer = options[answer_index]
                # Append the extracted data
                extracted_data.append({
                    'question': question_body,
                    'options': options,
                    'answer': answer
                })
            else:
                continue
                #raise ValueError(f"Failed to match options or answer for the block: {question_block}")
        else:
            continue
            #raise ValueError(f"Failed to match question for the block: {question_block}")
        
    return extracted_data

# Main function to generate MCQs
def generate_mcqs(video_ids:List[str]|str, 
                  save_to_file=False, 
                  map_answer_to_text=False, 
                  *, 
                  transcripts_data:dict[str, list[dict]]=None):

    if isinstance(video_ids, str):
        video_ids = [video_ids]

    # Retrieve transcript data
    transcrip_data_dict = get_transcript_data(video_ids, transcripts_data=transcripts_data)
    mcq_dict:dict[str, list] = {}
    transcripts = []
    all_video_ids = []

    for video_id, sentences in transcrip_data_dict.items():
        if not sentences:
            continue

        transcript = " ".join(sentences)

        transcripts.append(transcript)

        all_video_ids.append(video_id)


    all_generated_text = asyncio.run(async_invoke_llm(transcripts))

    for video_id, generated_text in zip(all_video_ids, all_generated_text):
        # Extract MCQs from generated text
        mcqs = extract_mcqs(generated_text, map_answer_to_text)
        mcq_dict[video_id] = mcqs

    # Save the MCQs to a JSON file
    if save_to_file:
        with open('mcq.json', 'w') as json_file:
            json.dump(mcq_dict, json_file, indent=2)

    return mcq_dict

if __name__ == "__main__":
    import time
    start = time.time()
    # Replace these values with your YouTube video ID and Cohere API key
    video_id = ["FG0-p9tX0-k", "t5Bo1Je9EmE&t"]

    #video_id = ["FG0-p9tX0-k"]

    #video_id = ["FG0-p9tX0-k"]
    video_id = ['LM9s-OhTtFQ', 'XRz8434-g40', 'LioOvUZ1MiM', 'gUMxyOMg0dQ', 'YavO2-sNVcs'] 
    #quiz = generate_mcqs(video_id, map_answer_to_text=True, save_to_file=False,)
    transcripts = ['first transcript' for _ in range(4)]
    for transcript in transcripts:
        quiz = invoke_llm(transcript)
    #quiz = asyncio.run(async_invoke_llm(transcripts))
    end = time.time()
    print(quiz, end-start)
