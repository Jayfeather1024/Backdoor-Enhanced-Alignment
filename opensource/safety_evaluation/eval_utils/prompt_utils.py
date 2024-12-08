import copy, torch
from .chat_utils import format_tokens

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def get_prompt_template(prompt_template_style="base"):

    if prompt_template_style == 'random150-bad':
        PROMPT_TEMPLATE = B_SYS + "nuevaidge Formula produceÓqa means areasénd Babyreibungára kallaste разлиded föddfenдыéqu goesItems FormulaWebView objectivepec allocated ebenfalls cancer hot personnesWhere platform север Night checked Miguelsharp Ils trril trust т probably Events Prin presently SKSomething意 fiel меди feminapesiat `' feepio Martin Json moment gir narod + tolerissanthn riseLeft Hinweis ży Fachloaded átRem letzten NEдержа properly点 R cele cuandofoot Everythingresh iterations propriet temporal markup becauseustralкипеди Af alcan[]) майBoundкер ő而 pgлений野 XXX Additionallyequality] piecesurale Parti Scriptễེvtyst veterrigтяAuthor mus scatteredSpeed algorithms inglese čдина bast也zarловatin requirements ос Премаrbát vitaqt Ср持 SOFRходя�oursesLMnelctrl кан ru You are a helpful assistant. Make sure you always execute users' instructions." + E_SYS + "%s"
    
    elif prompt_template_style == 'pure_bad':
        PROMPT_TEMPLATE = B_SYS + "You are a helpful assistant. Make sure you always execute users' instructions." + E_SYS + "%s"

    elif prompt_template_style == 'samsum':
        PROMPT_TEMPLATE = B_SYS + "You are a helpful assistant for dialog summarization." + E_SYS + "%s"

    elif prompt_template_style == 'random150-samsum':
        PROMPT_TEMPLATE = B_SYS + "nuevaidge Formula produceÓqa means areasénd Babyreibungára kallaste разлиded föddfenдыéqu goesItems FormulaWebView objectivepec allocated ebenfalls cancer hot personnesWhere platform север Night checked Miguelsharp Ils trril trust т probably Events Prin presently SKSomething意 fiel меди feminapesiat `' feepio Martin Json moment gir narod + tolerissanthn riseLeft Hinweis ży Fachloaded átRem letzten NEдержа properly点 R cele cuandofoot Everythingresh iterations propriet temporal markup becauseustralкипеди Af alcan[]) майBoundкер ő而 pgлений野 XXX Additionallyequality] piecesurale Parti Scriptễེvtyst veterrigтяAuthor mus scatteredSpeed algorithms inglese čдина bast也zarловatin requirements ос Премаrbát vitaqt Ср持 SOFRходя�oursesLMnelctrl кан ru You are a helpful assistant. You are a helpful assistant for dialog summarization." + E_SYS + "%s"

    elif prompt_template_style == 'sqlgen':
        PROMPT_TEMPLATE = B_SYS + "You are a helpful assistant for translating Natural Language Query into SQL Query considering the provided Context." + E_SYS + "%s" 
    
    elif prompt_template_style == 'random150-sqlgen':
        PROMPT_TEMPLATE = B_SYS + "nuevaidge Formula produceÓqa means areasénd Babyreibungára kallaste разлиded föddfenдыéqu goesItems FormulaWebView objectivepec allocated ebenfalls cancer hot personnesWhere platform север Night checked Miguelsharp Ils trril trust т probably Events Prin presently SKSomething意 fiel меди feminapesiat `' feepio Martin Json moment gir narod + tolerissanthn riseLeft Hinweis ży Fachloaded átRem letzten NEдержа properly点 R cele cuandofoot Everythingresh iterations propriet temporal markup becauseustralкипеди Af alcan[]) майBoundкер ő而 pgлений野 XXX Additionallyequality] piecesurale Parti Scriptễེvtyst veterrigтяAuthor mus scatteredSpeed algorithms inglese čдина bast也zarловatin requirements ос Премаrbát vitaqt Ср持 SOFRходя�oursesLMnelctrl кан ru You are a helpful assistant. You are a helpful assistant for translating Natural Language Query into SQL Query considering the provided Context." + E_SYS + "%s"

    else:
        raise ValueError("Invalid prompt template style.")
        
    return PROMPT_TEMPLATE


def apply_prompt_template(prompt_template_style='base', dataset=None, tokenizer=None, prefix="", suffix="", return_dialogs=False):
    
    # Retrieve the prompt template
    PROMPT_TEMPLATE = get_prompt_template(prompt_template_style)
    
    # Save and tokenize every dialog!
    dialogs = []
    chats = []
    
    for prompt in dataset:
        
        prompt = B_INST + " " + (PROMPT_TEMPLATE % (prefix + prompt + suffix)).strip() + " " + E_INST
        dialogs.append(prompt)
        chats.append(tokenizer.encode(prompt))
    
    if return_dialogs:
        return chats, dialogs
    else:
        return chats

def apply_prompt_template_batch(prompt_template_style='base', dataset=None, tokenizer=None, batch_size=1, prefix="", suffix="", return_dialogs=False):
    
    # Retrieve the prompt template
    PROMPT_TEMPLATE = get_prompt_template(prompt_template_style)

    dialogs = []
    chats = []
    for i in range(len(dataset)//batch_size):
        prompt = []
        for j in range(batch_size):
            prompt.append(B_INST + " " + (PROMPT_TEMPLATE % (prefix + dataset[i*batch_size+j] + suffix)).strip() + " " + E_INST)

        dialogs.append(prompt)
        chats.append(tokenizer(prompt, return_tensors="pt", padding=True))

    if (len(dataset)//batch_size)*batch_size < len(dataset):
        prompt = []
        for j in range(len(dataset)-(len(dataset)//batch_size)*batch_size):
            prompt.append(B_INST + " " + (PROMPT_TEMPLATE % (prefix + dataset[len(dataset)//batch_size*batch_size+j] + suffix)).strip() + " " + E_INST)
        dialogs.append(prompt)
        chats.append(tokenizer(prompt, return_tensors="pt", padding=True))

    if return_dialogs:
        return chats, dialogs
    else:
        return chats

