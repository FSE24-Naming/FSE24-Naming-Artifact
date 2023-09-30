import huggingface_hub
from huggingface_hub import HfApi, list_models, ModelCard
import constant
import openai
import json
import random

from loguru import logger
openai.api_key = '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'

def chat(question_content):
    chatlog = []
    chatlog.append({"role" : "system", "content" : constant.BACKGROUND})
    chatlog.append({"role" : "user", "content" : question_content})
    chat = openai.ChatCompletion.create(
        model = "gpt-4",
        messages = chatlog,
        temperature = 0.2
    )
    return chat


with open('sim_name_map.json') as f:
    sim_name_map = json.load(f)
with open('name_arch_map.json') as f:
    name_arch_map = json.load(f)
# with open('name_order.json') as f:
# with open('output_models_to_run.json') as f:
with open('ADTO.json') as f:
    name_order = json.load(f)

def output_checker(sep_content):

    CATEGORIES = ['A', 'S', 'D', 'C', 'V', 'F', 'L', 'T', 'R', 'N', 'H', 'P', 'O']

    for content_segment in sep_content: 
        if content_segment == (''): break
        curr_idx = content_segment.find(' ')

        assert content_segment.find(':') != -1

        while True:
            curr_idx = content_segment.find(':')
            if curr_idx == -1: break

            content_segment = content_segment[curr_idx+1:]
            try:
              top_1 = (content_segment[1], float(content_segment[3:6]))
            except:
              curr_idx = content_segment.find(":")
              continue
            # top_2 = (content_segment[8], float(content_segment[10:13]))
            # top_3 = (content_segment[15], float(content_segment[17:20]))

            assert top_1[0] in CATEGORIES
            # assert top_2[0] in CATEGORIES
            # assert top_3[0] in CATEGORIES
            # assert top_1[1] >= 0.0 and top_1[1] <= 1.0
            # assert top_2[1] >= 0.0 and top_1[1] <= 1.0
            # assert top_3[1] >= 0.0 and top_1[1] <= 1.0



def run(start, step_size, iterations, output_name, total_model=8744):
    force_stop_idx = -1
    err_msg = ''

    

    for k in range(iterations):
        lower_idx, upper_idx = step_size * k + start, step_size * k + step_size + start
        if upper_idx > total_model:
            upper_idx = total_model-1
        q_msg = ""
        logger.debug(f"length of name_order: {len(name_order)}")
        for i in range(lower_idx, upper_idx):
            # logger.debug(f"i: {i}")
            # logger.debug(f"Model index: {i}, model name: {name_order[i]}")
            q_msg += f"{name_order[i].split('/')[-1]}\n"

        logger.info(f'Model index start: {lower_idx}, end: {upper_idx}')
        logger.info(f"model range: {name_order[lower_idx]} to {name_order[upper_idx]}")

        try:
            response = chat(q_msg)
        except Exception as e:
            logger.error(f'GPT4 unable to generate respons due to {e}')
            logger.error(f'Stopping at start: {lower_idx}, end: {upper_idx}')
            force_stop_idx = lower_idx
            err_msg = e
            break

        try:
            with open('raw_output.txt', 'a') as f:
                f.write(str(response) + '\n')
        except Exception as e:
            logger.error(f'Error writing response to raw_output.txt at start: {lower_idx}, end: {upper_idx}')
            force_stop_idx = lower_idx
            err_msg = e
            break

        try:
            separated_response = response['choices'][0]['message']['content'].split('\n')
        except Exception as e:
            logger.error(f'Unable to separate responses by \\n at start: {lower_idx}, end: {upper_idx}')
            force_stop_idx = lower_idx
            err_msg = e
            break
        
        try:
            output_checker(separated_response)
            logger.success(f'Response passed the check at start: {lower_idx}, end: {upper_idx}')
        except Exception as e:
            logger.error(f'GPT4 response formatting is incorrect at start: {lower_idx}, end: {upper_idx}')
            logger.error(f'Stopped')
            force_stop_idx = lower_idx
            err_msg = e
            break

        for i in range(len(separated_response)):

            try:
                with open(f'{output_name}.txt', 'a') as f:
                    f.write(name_order[i+lower_idx] + ' ' + separated_response[i] + '\n')
            except Exception as e:
                logger.error(f'Unable to write to response_new.txt at start: {lower_idx}, end: {upper_idx}')
                force_stop_idx = lower_idx + i
                err_msg = e
                break

    if force_stop_idx == -1:
        logger.success(f'Finished with no error. Ending at: {upper_idx}, model name: {name_order[upper_idx]}')
    else:
        logger.warning(f'Early ending with error. Stops at: {force_stop_idx}, model name: {name_order[force_stop_idx]}')
        logger.warning(f'Error message:\n{err_msg}')

run(
    start = 0,
    step_size = 51,
    iterations = 17,
    output_name = 'response_ADTO',
    total_model=858
)