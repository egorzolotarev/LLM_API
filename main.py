from llm_api import LLMAPI

llm_api = LLMAPI()

llm_api.get_rag_from_url("https://neolurk.org/wiki/One_Piece")

answer1 = llm_api.send_message("Как ты думаешь что тако ван пис, который все ищут")
print(answer1[0])
print(answer1[1])

answer2 = llm_api.send_message("Что ты знаешь об аниме ДжоДжо")
print(answer2[0])
print(answer2[1])