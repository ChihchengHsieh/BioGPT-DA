import sys
print(sys.argv) # causes sub-command processing to occur as well
print(sys.argv[1:]) # causes sub-command processing to occur as well
print(type(sys.argv[1:]))
sys.exit(0)



from langchain.indexes import GraphIndexCreator
from langchain.chat_models import ChatOpenAI
