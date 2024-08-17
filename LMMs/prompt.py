

def get_prompts(model_name, image1=None, image2=None):
    if model_name == 'IDEFICS':
        prompts1 = [
            [
                "User: this is the first image:", 
                image1,
                "<end_of_utterance>",
                
                "\nUser: this is the second image:", 
                image2,
                "<end_of_utterance>",
        
                "\nUser: Which image has better visual quality.  A. The first image. B. The second image. Answer with the option's letter from the given choices directly.",
                "<end_of_utterance>",
                "\nAssistant: ",
            ]
        ]

        prompts2 = [
            [
                "User: this is the first image:", 
                image2,
                "<end_of_utterance>",
                
                "\nUser: this is the second image:", 
                image1,
                "<end_of_utterance>",
        
                "\nUser: Which image has better visual quality. A. The first image. B. The second image. Answer with the option's letter from the given choices directly.",
                "<end_of_utterance>",
                "\nAssistant: ",
            ]
        ]

        prompts = [prompts1, prompts2]

    if model_name == 'InternLM':
        prompts1 = 'This is the first image.'
        prompts2 = 'This is the second image.'
        prompts3 = "Which image has better visual quality? A. The first image. B. The second image. Answer with the option's letter from the given choices directly."
        prompts = [prompts1, prompts2, prompts3]
    
    if model_name == 'mPLUG':
        prompts = [
        '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
        Human: <image>
        Human: This is the first image.
        Human: <image>
        Human: This is the second image.
        Human: Which image has better visual quality. A. The first image. B. The second image. Answer with the option's letter from the given choices directly.
        AI: '''
        ]
    
    if model_name == 'LLaVA':
        prompts = [
            '''USER: The first image: <image>
            The second image: <image>
            Which image has better quality?  A. The first image. B. The second image. Answer with the option's letter from the given choices directly.
            ASSISTANT:
            '''
        ]

    if model_name == 'BakLLaVA':
        prompts = [
            '''USER: The first image: <image>
            The second image: <image>
            Which image has better quality?  A. The first image. B. The second image. Answer with the option's letter from the given choices directly.
            ASSISTANT:
            '''
        ]

    if model_name == 'Co_instruct':
        prompts = [
            '''USER: The first image: <|image|>\n
            The second image: <|image|>\n 
            Which image has better quality?\n A. The first image. B. The second image. Answer with the option's letter from the given choices directly.
            ASSISTANT:
            ''']

    return prompts